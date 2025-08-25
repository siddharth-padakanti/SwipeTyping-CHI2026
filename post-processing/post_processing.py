import os
import math
import ast
import re
import numpy as np
import pandas as pd
import cv2
from tkinter import Tk, filedialog
from datetime import datetime

# =========================
# Tunables
# =========================
MAX_GAP_MS       = 400.0
TAU_RANGE_MS     = 1500
TAU_GRID_MS      = 20

RANSAC_REPROJ    = 3.0
MIN_INLIERS      = 40

SWITCH_PENALTY   = 140.0
SESSION_PRIOR    = 70.0
BOOK_PRIOR       = 6.0
HAND_SOFT        = 100.0
NON_WL_PEN       = 220.0
EDGE_RING_PEN    = 120.0
ANCHOR_PX_W      = 0.35

MOTION_HALF_MS   = 180
MOTION_STEP_MS   = 50
MOTION_WEIGHT    = 80.0

FORCE_FINGERS = None  # e.g. ["Left_Index","Right_Index"]

ALL_FINGERS = [
    "Left_Pinky","Left_Ring","Left_Middle","Left_Index","Left_Thumb",
    "Right_Thumb","Right_Index","Right_Middle","Right_Ring","Right_Pinky"
]
LEFT_SET  = {"Left_Pinky","Left_Ring","Left_Middle","Left_Index","Left_Thumb"}
RIGHT_SET = {"Right_Pinky","Right_Ring","Right_Middle","Right_Index","Right_Thumb"}

LEFT_KEYS  = set(list("qwertasdfgzxcvb"))
RIGHT_KEYS = set(list("yuiophjklnm,."))
BOOK_DEFAULT = {
    **{k:"Left Pinky"  for k in "qaz"},
    **{k:"Left Ring"   for k in "wsx"},
    **{k:"Left Middle" for k in "edc"},
    **{k:"Left Index"  for k in "rfvtgb"},
    **{k:"Right Index"  for k in "yhnujm"},
    **{k:"Right Middle" for k in "ik"},
    **{k:"Right Ring"   for k in "ol"},
    **{k:"Right Pinky"  for k in "p,."},
}
RIGHT_EDGE_KEYS = {"o","p",",","."}

# =========================
# Parsers (robust)
# =========================
_num_re = re.compile(r"[-+]?\d*\.?\d+")

def parse_time(ts):
    s = str(ts).strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return datetime.strptime(s.split()[0], "%H:%M:%S")

def parse_xy(text):
    t = str(text).strip()
    if t in ("-", "", "[]", "null", "None"): return float("nan"), float("nan")
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    nums = _num_re.findall(t)
    if len(nums) < 2: return float("nan"), float("nan")
    try:
        return float(nums[0]), float(nums[1])
    except Exception:
        return float("nan"), float("nan")

def parse_num_list(cell):
    if cell is None: return []
    s = str(cell).strip()
    if s in ("-", "", "[]", "null", "None"): return []
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    nums = _num_re.findall(s)
    try:
        return [float(n) for n in nums]
    except Exception:
        return []

def parse_keys_field(cell):
    if cell is None:
        return [None]
    s = str(cell).strip()
    if s in ("-", "", "[]", "null", "None"):
        return [None]
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            out = []
            for x in val:
                if x is None: out.append(None)
                else:
                    xs = str(x).strip().lower()
                    out.append(None if xs in ("null","none","-","") else xs)
            return out if out else [None]
        elif isinstance(val, str):
            xs = val.strip().lower()
            return [None] if xs in ("null","none","-","") else [xs]
    except Exception:
        pass
    return [s.lower()]

def key_hand(letter: str):
    if not letter: return None
    ch = letter.strip().lower()
    if ch in LEFT_KEYS:  return "Left"
    if ch in RIGHT_KEYS: return "Right"
    return None

def default_finger_for_key(letter: str):
    ch = (letter or "").strip().lower()
    return BOOK_DEFAULT.get(ch)

def most_common(lst):
    if not lst: return None
    vals = [x for x in lst if x and x != "Unknown"]
    if not vals: return None
    return pd.Series(vals).mode().iloc[0]

# =========================
# Time & interpolation
# =========================
def to_ms(t0, ser): return (ser - t0).dt.total_seconds() * 1000.0

def interp_finger_at(times_ms, xs, ys, t_ms, max_gap_ms=MAX_GAP_MS):
    idx = np.searchsorted(times_ms, t_ms, side="left")
    i0 = idx - 1
    while i0 >= 0 and (np.isnan(xs[i0]) or np.isnan(ys[i0])): i0 -= 1
    i1 = idx; n = len(times_ms)
    while i1 < n and (np.isnan(xs[i1]) or np.isnan(ys[i1])): i1 += 1
    if i0 < 0 and i1 >= n: return float("nan"), float("nan")
    if i0 < 0:
        if abs(times_ms[i1]-t_ms) > max_gap_ms: return float("nan"), float("nan")
        return xs[i1], ys[i1]
    if i1 >= n:
        if abs(t_ms-times_ms[i0]) > max_gap_ms: return float("nan"), float("nan")
        return xs[i0], ys[i0]
    dt = times_ms[i1] - times_ms[i0]
    if dt <= 0:
        return (xs[i0], ys[i0]) if abs(t_ms-times_ms[i0]) <= abs(times_ms[i1]-t_ms) else (xs[i1], ys[i1])
    if (t_ms - times_ms[i0]) > max_gap_ms and (times_ms[i1] - t_ms) > max_gap_ms:
        return float("nan"), float("nan")
    a = (t_ms - times_ms[i0]) / dt
    return xs[i0] + a*(xs[i1]-xs[i0]), ys[i0] + a*(ys[i1]-ys[i0])

def motion_energy(times_ms, xs, ys, center_ms, half_win_ms, step_ms):
    if not np.isfinite(center_ms): return 0.0
    ts = np.arange(center_ms - half_win_ms, center_ms + half_win_ms + 1, step_ms, dtype=float)
    pts = []
    for t in ts:
        fx, fy = interp_finger_at(times_ms, xs, ys, t)
        pts.append(None if (np.isnan(fx) or np.isnan(fy)) else (fx, fy))
    e = 0.0; prev = None
    for p in pts:
        if p is None: prev = None; continue
        if prev is not None: e += math.hypot(p[0]-prev[0], p[1]-prev[1])
        prev = p
    return e

# =========================
# Vision: video→UI homography
# =========================
def grab_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, n//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def find_homography_v2ui(video_frame_bgr, ui_bgr):
    if video_frame_bgr is None or ui_bgr is None: return None
    img1 = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(ui_bgr,          cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < MIN_INLIERS: return None
    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_REPROJ)
    if H is None: return None
    inliers = int(mask.sum())
    return H if inliers >= MIN_INLIERS else None

def perspective_transform_points(H, xy_array):
    pts = xy_array.reshape(-1,1,2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return out

# =========================
# Main
# =========================
root = Tk(); root.withdraw()

gesture_path = filedialog.askopenfilename(title="Select Gesture Log CSV", filetypes=[("CSV Files","*.csv")])
finger_path  = filedialog.askopenfilename(title="Select Finger Log CSV",   filetypes=[("CSV Files","*.csv")])
video_path   = filedialog.askopenfilename(title="Select Warped Screen Video (MP4/AVI)", filetypes=[("Video","*.mp4 *.avi *.mov *.mkv")])
ui_image     = filedialog.askopenfilename(title="Select UI Screenshot (PNG/JPG)", filetypes=[("Images","*.png *.jpg *.jpeg")])

base = os.path.basename(gesture_path); name, _ = os.path.splitext(base)
participant_id = name.rsplit("__", 1)[-1]

gest = pd.read_csv(gesture_path)
fing = pd.read_csv(finger_path)

finger_cols = ALL_FINGERS if FORCE_FINGERS is None else FORCE_FINGERS[:]

# --- times ---
gest["t"] = gest["Time"].apply(parse_time)
fing["t"] = fing["Timestamp"].apply(parse_time)
t0 = min(gest["t"].min(), fing["t"].min())
gest["tms"] = to_ms(t0, gest["t"])
fing["tms"] = to_ms(t0, fing["t"])

# --- detect gesture cols (incl. EndX/EndY/ProjectKey) ---
def _find_col(df, candidates):
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in norm: return norm[cand]
    for k, orig in norm.items():
        if any(cand in k for cand in candidates): return orig
    return None

X_CANDS     = ["x","gx","x_ui","ui_x","uix","x(px)","x (px)","x_ui(px)","posx","clientx","screenx","tapx","startx"]
Y_CANDS     = ["y","gy","y_ui","ui_y","uiy","y(px)","y (px)","y_ui(px)","posy","clienty","screeny","tapy","starty"]
ENDX_CANDS  = ["endx","x2","x_end","end_x","projectx","projx","end x"]
ENDY_CANDS  = ["endy","y2","y_end","end_y","projecty","projy","end y"]
KEYS_CANDS  = ["keys","key","letters","chars","labels","gesturekeys","key_list","keystrokes","tapkey","startkey","endkey"]
PROJK_CANDS = ["projectkey","projectedkey","projkey","predkey","predictedkey","endkey"]

x_col    = _find_col(gest, X_CANDS)
y_col    = _find_col(gest, Y_CANDS)
endx_col = _find_col(gest, ENDX_CANDS)
endy_col = _find_col(gest, ENDY_CANDS)
keys_col = _find_col(gest, KEYS_CANDS)
projk_col= _find_col(gest, PROJK_CANDS)

if x_col is None or y_col is None:
    maybe_xy = None
    for c in gest.columns:
        sample = str(gest[c].dropna().astype(str).head(20).tolist())
        if "[" in sample and "," in sample and "]" in sample:
            maybe_xy = c; break
    if maybe_xy:
        gest["gx_ui"] = gest[maybe_xy].apply(lambda s: parse_num_list(s)[-1] if parse_num_list(s) else float("nan")).astype(float)
        gest["gy_ui"] = gest[maybe_xy].apply(lambda s: parse_num_list(s)[-1] if parse_num_list(s) else float("nan")).astype(float)
        x_col = y_col = maybe_xy
    else:
        raise ValueError("Couldn't find gesture X/Y columns.")
else:
    gest["gx_ui"] = gest[x_col].apply(lambda s: (parse_num_list(s) or [float('nan')])[-1]).astype(float)
    gest["gy_ui"] = gest[y_col].apply(lambda s: (parse_num_list(s) or [float('nan')])[-1]).astype(float)

if keys_col is None:
    for c in gest.columns:
        sample = gest[c].dropna().astype(str).head(20).str.lower().str.cat(sep=" ")
        if "[" in sample and "]" in sample and any(k in sample for k in list("abcdefghijklmnopqrstuvwxyz")):
            keys_col = c; break
if keys_col is None:
    raise ValueError("Couldn't find a keys-like column.")

g_ui = gest[["gx_ui","gy_ui"]].to_numpy()
t_g  = gest["tms"].to_numpy()

# --- expand finger columns (warped plane, 640x480) ---
for col in finger_cols:
    arr = fing[col].apply(parse_xy)
    fing[f"{col}_x"] = [p[0] for p in arr]
    fing[f"{col}_y"] = [p[1] for p in arr]
times_ms = fing["tms"].to_numpy()

# --- video→UI homography; fallback to fixed scale ---
H_v2ui = None
try:
    if video_path and ui_image:
        frame = grab_middle_frame(video_path)
        ui_img = cv2.imread(ui_image)
        if frame is not None and ui_img is not None:
            H_v2ui = find_homography_v2ui(frame, ui_img)
except Exception:
    H_v2ui = None

if H_v2ui is None:
    sx, sy = 880.0/640.0, 320.0/480.0
    H_v2ui = np.array([[sx, 0, 0],
                       [0, sy, 0],
                       [0,  0, 1]], dtype=np.float32)

# --- tiny clock-skew search (UI plane) ---
def median_distance_for_tau(tau_ms):
    dists = []
    for i in range(len(gest)):
        xg, yg = g_ui[i]
        if not np.isfinite(xg) or not np.isfinite(yg): continue
        keys_list = parse_keys_field(gest.at[i, keys_col])
        key_last  = next((k for k in reversed(keys_list) if k), None)
        h = key_hand(key_last)
        for col in finger_cols:
            if h == "Left" and col not in LEFT_SET:  continue
            if h == "Right" and col not in RIGHT_SET: continue
            xs = fing[f"{col}_x"].to_numpy(); ys = fing[f"{col}_y"].to_numpy()
            fx, fy = interp_finger_at(times_ms, xs, ys, t_g[i] + tau_ms)
            if np.isnan(fx) or np.isnan(fy): continue
            ui_xy = perspective_transform_points(H_v2ui, np.array([[fx,fy]], dtype=np.float32))[0]
            dists.append(math.hypot(ui_xy[0]-xg, ui_xy[1]-yg))
    return float(np.median(dists)) if dists else float("inf")

best_tau, best_med = 0.0, float("inf")
for tau in range(-TAU_RANGE_MS, TAU_RANGE_MS+1, TAU_GRID_MS):
    m = median_distance_for_tau(tau)
    if m < best_med:
        best_tau, best_med = float(tau), m

# --- distances in the same (UI) plane ---
N = len(gest); F = len(finger_cols)
D = np.full((N,F), 1e6, dtype=float)
for j, col in enumerate(finger_cols):
    xs = fing[f"{col}_x"].to_numpy(); ys = fing[f"{col}_y"].to_numpy()
    for i in range(N):
        fx, fy = interp_finger_at(times_ms, xs, ys, t_g[i] + best_tau)
        if np.isnan(fx) or np.isnan(fy): continue
        ui_xy = perspective_transform_points(H_v2ui, np.array([[fx,fy]], dtype=np.float32))[0]
        D[i,j] = math.hypot(ui_xy[0]-g_ui[i,0], ui_xy[1]-g_ui[i,1])

# --- session whitelist & anchors (in UI plane) ---
key_counts = {}
coarse = []
for i in range(N):
    key_last = next((k for k in parse_keys_field(gest.at[i, keys_col])[::-1] if k), None)
    if not key_last: coarse.append(None); continue
    candidates = [(D[i,j], finger_cols[j]) for j in range(F) if D[i,j] < 1e6]
    if not candidates: coarse.append(None); continue
    best = min(candidates, key=lambda x: x[0])[1]
    coarse.append(best)
    key_counts.setdefault(key_last.lower(), {})
    key_counts[key_last.lower()][best] = key_counts[key_last.lower()].get(best, 0) + 1

key_whitelist = {
    k: [f for f,_ in sorted(cnts.items(), key=lambda kv: kv[1], reverse=True)[:2]]
    for k, cnts in key_counts.items()
}
session_prior = {
    k: (sorted(cnts.items(), key=lambda kv: kv[1], reverse=True)[0][0].replace("_"," "))
    for k, cnts in key_counts.items()
}

anchors = {}
for i in range(N):
    key_last = next((k for k in parse_keys_field(gest.at[i, keys_col])[::-1] if k), None)
    bestf = coarse[i]
    if not key_last or not bestf: continue
    xs = fing[f"{bestf}_x"].to_numpy(); ys = fing[f"{bestf}_y"].to_numpy()
    fx, fy = interp_finger_at(times_ms, xs, ys, t_g[i] + best_tau)
    if np.isnan(fx) or np.isnan(fy): continue
    ui_xy = perspective_transform_points(H_v2ui, np.array([[fx,fy]], dtype=np.float32))[0]
    anchors.setdefault((key_last.lower(), bestf), []).append(tuple(ui_xy))
for kf, pts in anchors.items():
    arr = np.array(pts)
    anchors[kf] = (float(np.median(arr[:,0])), float(np.median(arr[:,1])))

xvals = g_ui[:,0]; xvals = xvals[np.isfinite(xvals)]
midline = float(np.median(xvals)) if xvals.size else 440.0

motion = np.zeros((N,F), dtype=float)
for j, col in enumerate(finger_cols):
    xs = fing[f"{col}_x"].to_numpy(); ys = fing[f"{col}_y"].to_numpy()
    for i in range(N):
        motion[i,j] = motion_energy(times_ms, xs, ys, t_g[i] + best_tau, MOTION_HALF_MS, MOTION_STEP_MS)
row_max = motion.max(axis=1, keepdims=True); row_max[row_max==0] = 1.0
motion_pen = (1.0 - motion/row_max) * MOTION_WEIGHT

canon_pen = np.zeros((N,F)); book_pen = np.zeros((N,F))
hand_pen  = np.zeros((N,F)); white_pen= np.zeros((N,F)); anch_pen = np.zeros((N,F))
for i in range(N):
    key_last = next((k for k in parse_keys_field(gest.at[i, keys_col])[::-1] if k), None)
    sess = session_prior.get((key_last or "").lower())
    book = BOOK_DEFAULT.get((key_last or "").lower())
    expected_hand = "Left" if (np.isfinite(g_ui[i,0]) and g_ui[i,0] < midline) else ("Right" if np.isfinite(g_ui[i,0]) else None)
    wl = key_whitelist.get((key_last or "").lower(), [])
    for j, col in enumerate(finger_cols):
        nm = col.replace("_"," ")
        if sess and nm != sess: canon_pen[i,j] = SESSION_PRIOR
        if book and nm != book: book_pen[i,j] = BOOK_PRIOR
        if expected_hand == "Left" and col in RIGHT_SET: hand_pen[i,j] = HAND_SOFT
        if expected_hand == "Right" and col in LEFT_SET:  hand_pen[i,j] = HAND_SOFT
        if wl and col not in wl: white_pen[i,j] = NON_WL_PEN
        if key_last and key_last.lower() in RIGHT_EDGE_KEYS:
            if sess == "Right Middle" and col in {"Right_Ring","Right_Pinky"}:
                white_pen[i,j] += EDGE_RING_PEN
        if key_last and (key_last.lower(), col) in anchors and D[i,j] < 1e6:
            ax, ay = anchors[(key_last.lower(), col)]
            anch_pen[i,j] = ANCHOR_PX_W * math.hypot(ax - g_ui[i,0], ay - g_ui[i,1])

C = D + motion_pen + canon_pen + book_pen + hand_pen + white_pen + anch_pen
F = len(finger_cols)
Tmat = np.full((F,F), SWITCH_PENALTY); np.fill_diagonal(Tmat, 0.0)
dp = np.zeros((N,F)); bt = np.zeros((N,F), dtype=int)
dp[0,:] = C[0,:]
for i in range(1,N):
    prev = dp[i-1,:].reshape(-1,1) + Tmat
    bt[i,:] = prev.argmin(axis=0)
    dp[i,:] = C[i,:] + prev.min(axis=0)
path = np.zeros(N, dtype=int)
path[-1] = dp[-1,:].argmin()
for i in range(N-2,-1,-1):
    path[i] = bt[i+1, path[i+1]]
gest["Finger"] = [finger_cols[k].replace("_"," ") for k in path]

# =========================
# Fix Keys and FORMAT OUTPUT lists (+ EndX/EndY/ProjectKey)
# =========================
raw_keys_series = gest[keys_col]
type_col = "Type" if "Type" in gest.columns else None
type_vals = gest[type_col] if type_col else pd.Series(["tap"]*len(gest))

def concrete_tap_key_at(i):
    arr = parse_keys_field(raw_keys_series.iloc[i])
    return arr[0] if len(arr) == 1 and arr[0] else None

def nearest_prev_tap(i):
    j = i - 1
    while j >= 0:
        k = concrete_tap_key_at(j)
        if k: return k
        j -= 1
    return None

def nearest_next_tap(i):
    j = i + 1; n = len(raw_keys_series)
    while j < n:
        k = concrete_tap_key_at(j)
        if k: return k
        j += 1
    return None

def first_non_null(lst):
    for x in lst:
        if x: return x
    return None

def last_non_null(lst):
    for x in reversed(lst):
        if x: return x
    return None

def fmt_keys_list_plain(lst):
    lst = [k for k in lst if k]
    return "[" + ", ".join(lst) + "]" if lst else "[]"

def fmt_num_list_str_from_vals(vals):
    if not vals: return "[]"
    out = []
    for v in vals:
        if abs(v - round(v)) < 0.5:
            out.append(str(int(round(v))))
        else:
            s = f"{v:.3f}".rstrip("0").rstrip(".")
            out.append(s)
    return "[" + ", ".join(out) + "]"

# merge helpers to append EndX/EndY if present
def merged_xy_list(row_cell, end_cell):
    vals = parse_num_list(row_cell)
    endv = None
    if end_cell is not None:
        ends = parse_num_list(end_cell)
        if ends:
            endv = ends[-1]
    if endv is not None:
        if len(vals) == 0:
            vals = [endv]
        elif len(vals) == 1 and abs(vals[-1] - endv) > 1e-6:
            vals = [vals[0], endv]
        elif len(vals) >= 2:
            pass  # already has start,end
    return vals

# Build X/Y/Keys with EndX/EndY/ProjectKey merged
X_out_str, Y_out_str, keys_out_str = [], [], []
for i in range(len(gest)):
    row_type = str(type_vals.iloc[i]).strip().lower()

    # X/Y: append EndX/EndY if available
    x_vals = merged_xy_list(gest.at[i, x_col], gest.at[i, endx_col] if endx_col else None)
    y_vals = merged_xy_list(gest.at[i, y_col], gest.at[i, endy_col] if endy_col else None)

    # ensure taps stay single-element
    if row_type != "swipe":
        if len(x_vals) >= 1: x_vals = [x_vals[0]]
        if len(y_vals) >= 1: y_vals = [y_vals[0]]

    X_out_str.append(fmt_num_list_str_from_vals(x_vals))
    Y_out_str.append(fmt_num_list_str_from_vals(y_vals))

    # Keys: enforce [start,end] for swipes; use ProjectKey for missing end
    arr = parse_keys_field(raw_keys_series.iloc[i])
    if row_type == "swipe":
        start_k = first_non_null(arr)
        end_k   = last_non_null(arr) if len(arr) > 1 else None

        # ProjectKey takes priority for missing end
        proj_k = None
        if projk_col:
            proj_raw = gest.at[i, projk_col]
            if pd.notna(proj_raw):
                ps = str(proj_raw).strip().lower()
                if ps not in ("", "none", "null", "-"): proj_k = ps

        if start_k is None:
            start_k = nearest_prev_tap(i) or nearest_next_tap(i)
        if end_k is None:
            end_k = proj_k or nearest_next_tap(i) or nearest_prev_tap(i) or start_k

        if start_k is None:
            finger = str(gest.at[i, "Finger"])
            start_k = "a" if finger.startswith("Left") else "l"
        if end_k is None:
            finger = str(gest.at[i, "Finger"])
            end_k = "a" if finger.startswith("Left") else "l"

        keys_out_str.append(fmt_keys_list_plain([start_k, end_k]))
    else:
        k = last_non_null(arr) or nearest_next_tap(i) or nearest_prev_tap(i)
        if k is None:
            finger = str(gest.at[i, "Finger"])
            k = "a" if finger.startswith("Left") else "l"
        keys_out_str.append(fmt_keys_list_plain([k]))

# =========================
# Save exactly as: Time,Type,X,Y,Keys,Finger
# =========================
out = pd.DataFrame({
    "Time":   gest["Time"],
    "Type":   type_vals,
    "X":      X_out_str,
    "Y":      Y_out_str,
    "Keys":   keys_out_str,
    "Finger": gest["Finger"]
})

now = datetime.now()
default_name = (
    f"gesture_finger_match_{now.month:02d}-{now.day:02d}-{now.year}"
    f"__{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    f"__{participant_id}.csv"
)

save_path = filedialog.asksaveasfilename(
    title="Save Result CSV",
    defaultextension=".csv",
    filetypes=[("CSV files","*.csv")],
    initialfile=default_name
)
if save_path:
    out.to_csv(save_path, index=False)
    print(f"Result saved to: {save_path}")
else:
    print("Save cancelled — no file written.")
