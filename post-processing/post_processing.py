import os
import math
import numpy as np
import pandas as pd
import cv2
from tkinter import Tk, filedialog
from datetime import datetime

# =============== Tunables ===============
# interpolation / timing
MAX_GAP_MS       = 400.0
TAU_RANGE_MS     = 1500
TAU_GRID_MS      = 20

# geometry / fitting
CAL_ITERS        = 2
RANSAC_REPROJ    = 3.0      # px
MIN_INLIERS      = 40

# costs
SWITCH_PENALTY   = 140.0
SESSION_PRIOR    = 70.0
BOOK_PRIOR       = 6.0
HAND_SOFT        = 100.0
NON_WL_PEN       = 220.0
EDGE_RING_PEN    = 120.0
ANCHOR_PX_W      = 0.35
DIST_REJECT_PX   = 90.0

# motion term
MOTION_HALF_MS   = 180
MOTION_STEP_MS   = 50
MOTION_WEIGHT    = 80.0

# restrict to a subset if you ran a constrained protocol
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

# =============== Parsers ===============
def parse_time(ts): return datetime.strptime(str(ts), "%H:%M:%S.%f")

def parse_gcoord(text):
    t = str(text).strip()
    if t in ("-", "", "[]"): return None
    parts = [p.strip() for p in t.strip("[]").split(",") if p.strip()]
    try: return float(parts[-1])
    except: return None

def parse_xy(text):
    t = str(text).strip()
    if t in ("-", "", "[]"): return float("nan"), float("nan")
    parts = [p.strip() for p in t.strip("[]").split(",") if p.strip()]
    if len(parts) != 2: return float("nan"), float("nan")
    try: return float(parts[0]), float(parts[1])
    except: return float("nan"), float("nan")

def parse_keys_field(cell):
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        raw = [x.strip() for x in s[1:-1].split(",")]
        out = []
        for x in raw:
            x = x.strip().strip("'").strip('"').lower()
            if x in ("null", "", "none", "-"):
                out.append(None)
            else:
                out.append(x)
        return out
    if s in ("-", "", "[]", "null", "None"): return [None]
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

# =============== Time helpers ===============
def to_ms(t0, ser): return (ser - t0).dt.total_seconds() * 1000.0

def interp_finger_at(times_ms, xs, ys, t_ms, max_gap_ms=MAX_GAP_MS):
    idx = np.searchsorted(times_ms, t_ms, side="left")
    i0 = idx - 1
    while i0 >= 0 and (np.isnan(xs[i0]) or np.isnan(ys[i0])): i0 -= 1
    i1 = idx
    n = len(times_ms)
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
        if np.isnan(fx) or np.isnan(fy): pts.append(None)
        else: pts.append((fx, fy))
    e = 0.0; prev = None
    for p in pts:
        if p is None: prev = None; continue
        if prev is not None: e += math.hypot(p[0]-prev[0], p[1]-prev[1])
        prev = p
    return e

# =============== Vision: video→UI homography ===============
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
    if d1 is None or d2 is None or len(k1)<20 or len(k2)<20:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    if len(good) < MIN_INLIERS:
        return None

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_REPROJ)
    if H is None: return None
    inliers = int(mask.sum())
    return H if inliers >= MIN_INLIERS else None

def perspective_transform_points(H, xy_array):
    # xy_array: (N,2)
    pts = xy_array.reshape(-1,1,2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return out

# =============== Main ===============
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

# --- parse times ---
gest["t"] = gest["Time"].apply(parse_time)
fing["t"] = fing["Timestamp"].apply(parse_time)
t0 = min(gest["t"].min(), fing["t"].min())
gest["tms"] = to_ms(t0, gest["t"])
fing["tms"] = to_ms(t0, fing["t"])

# --- parse gesture coords (UI plane) ---
gest["gx_ui"] = gest["X"].apply(parse_gcoord).astype(float)
gest["gy_ui"] = gest["Y"].apply(parse_gcoord).astype(float)
g_ui = gest[["gx_ui","gy_ui"]].to_numpy()
t_g  = gest["tms"].to_numpy()

# --- expand finger columns (warped plane, 640x480) ---
for col in finger_cols:
    arr = fing[col].apply(parse_xy)
    fing[f"{col}_x"] = [p[0] for p in arr]
    fing[f"{col}_y"] = [p[1] for p in arr]
times_ms = fing["tms"].to_numpy()

# --- compute video→UI homography (preferred) ---
H_v2ui = None
try:
    frame = grab_middle_frame(video_path)
    ui_img = cv2.imread(ui_image)
    if frame is not None and ui_img is not None:
        H_v2ui = find_homography_v2ui(frame, ui_img)
except Exception:
    H_v2ui = None

# Fallback: pure scale from 640x480 → 880x320
if H_v2ui is None:
    sx, sy = 880.0/640.0, 320.0/480.0
    H_v2ui = np.array([[sx, 0, 0],
                       [0, sy, 0],
                       [0,  0, 1]], dtype=np.float32)

# --- tiny clock-skew search (in UI plane) ---
def median_distance_for_tau(tau_ms):
    dists = []
    for i in range(len(gest)):
        xg, yg = g_ui[i]
        if not np.isfinite(xg) or not np.isfinite(yg): continue
        keys_list = parse_keys_field(gest.at[i,"Keys"])
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
    key_last = next((k for k in parse_keys_field(gest.at[i,"Keys"])[::-1] if k), None)
    if not key_last: coarse.append(None); continue
    candidates = [(D[i,j], finger_cols[j]) for j in range(F) if D[i,j] < 1e6]
    if not candidates: coarse.append(None); continue
    best = min(candidates, key=lambda x: x[0])[1]
    coarse.append(best)
    key_counts.setdefault(key_last.lower(), {})
    key_counts[key_last.lower()][best] = key_counts[key_last.lower()].get(best, 0) + 1

key_whitelist = {k: [f for f,_ in sorted(cnts.items(), key=lambda kv: kv[1], reverse=True)[:2]]
                 for k, cnts in key_counts.items()}
session_prior = {k: (sorted(cnts.items(), key=lambda kv: kv[1], reverse=True)[0][0].replace("_"," "))
                 for k, cnts in key_counts.items()}

anchors = {}
for i in range(N):
    key_last = next((k for k in parse_keys_field(gest.at[i,"Keys"])[::-1] if k), None)
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

# --- hand midline from UI geometry ---
# use the gesture distribution directly
xvals = g_ui[:,0]; xvals = xvals[np.isfinite(xvals)]
midline = float(np.median(xvals)) if xvals.size else 440.0  # center of 880

# --- motion energy (still computed on finger stream, mapped per-row) ---
motion = np.zeros((N,F), dtype=float)
for j, col in enumerate(finger_cols):
    xs = fing[f"{col}_x"].to_numpy(); ys = fing[f"{col}_y"].to_numpy()
    for i in range(N):
        motion[i,j] = motion_energy(times_ms, xs, ys, t_g[i] + best_tau, MOTION_HALF_MS, MOTION_STEP_MS)
row_max = motion.max(axis=1, keepdims=True); row_max[row_max==0] = 1.0
motion_pen = (1.0 - motion/row_max) * MOTION_WEIGHT

# --- penalties ---
canon_pen = np.zeros((N,F)); book_pen = np.zeros((N,F))
hand_pen  = np.zeros((N,F)); white_pen= np.zeros((N,F)); anch_pen = np.zeros((N,F))

for i in range(N):
    key_last = next((k for k in parse_keys_field(gest.at[i,"Keys"])[::-1] if k), None)
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

# --- final cost & Viterbi ---
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

labels = [finger_cols[k].replace("_"," ") for k in path]
gest["Finger"] = labels

# --- repair swipe 'null' values ---
keys_fixed = []
single_keys = [(parse_keys_field(k)[0] if len(parse_keys_field(k))==1 else None) for k in gest["Keys"]]
def nearest_prev(i):
    j = i-1
    while j >= 0:
        k = single_keys[j]
        if k: return k
        j -= 1
    return None
def nearest_next(i):
    j = i+1; n = len(single_keys)
    while j < n:
        k = single_keys[j]
        if k: return k
        j += 1
    return None

for i, cell in enumerate(gest["Keys"]):
    arr = parse_keys_field(cell)
    if len(arr) == 1:
        keys_fixed.append(arr[0] if arr[0] is not None else (nearest_next(i) or nearest_prev(i) or "")); continue
    arr2 = arr[:]
    for j, val in enumerate(arr2):
        if val is not None: continue
        cand = None
        if j == len(arr2)-1: cand = nearest_next(i)
        if cand is None and j == 0: cand = nearest_prev(i)
        if cand is None:
            other = next((x for x in arr2 if x is not None), None)
            cand = other
        if cand is None:
            finger = gest.at[i, "Finger"]
            cand = ("a" if isinstance(finger,str) and finger.startswith("Left") else "l")
        arr2[j] = cand
    keys_fixed.append("[" + ", ".join(k for k in arr2 if k) + "]")
gest["Keys"] = keys_fixed

# --- save ---
out = gest[["Time","Type","X","Y","Keys","Finger"]].copy()
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
