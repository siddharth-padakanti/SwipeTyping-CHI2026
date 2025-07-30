import pandas as pd
from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import math

UI_W, UI_H       = 880.0, 320.0
WARP_W, WARP_H   = 640.0, 480.0
SCALE_X = WARP_W / UI_W   # ≈ 0.727
SCALE_Y = WARP_H / UI_H   # = 1.5

def parse_time(ts):
    return datetime.strptime(ts, "%H:%M:%S.%f")

def parse_gesture_coord(text):
    t = str(text).strip()
    if t in ("-", "", "[]"):
        return None
    parts = [p.strip() for p in t.strip("[]").split(",") if p.strip()]
    try:
        return float(parts[-1])
    except:
        return None

def parse_finger_coord(text):
    """
    Given a string like "[123,456]" or "-", return (123.0, 456.0),
    or None if missing / malformed.
    """
    t = str(text).strip()
    if t in ("-", "", "[]"):
        return None
    body = t.strip("[]")
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except:
        return None


root = Tk()
root.withdraw()

gesture_path = filedialog.askopenfilename(
    title="Select Gesture Log CSV", filetypes=[("CSV Files", "*.csv")]
)
finger_path = filedialog.askopenfilename(
    title="Select Finger Log CSV", filetypes=[("CSV Files", "*.csv")]
)

gest = pd.read_csv(gesture_path)
fing = pd.read_csv(finger_path)

gest["ParsedTime"] = gest["Time"].apply(parse_time)
gest["GX"] = gest["X"].apply(parse_gesture_coord)
gest["GY"] = gest["Y"].apply(parse_gesture_coord)

fing["ParsedTime"] = fing["Timestamp"].apply(parse_time)

finger_cols = [
    'Left_Pinky','Left_Ring','Left_Middle','Left_Index','Left_Thumb',
    'Right_Thumb','Right_Index','Right_Middle','Right_Ring','Right_Pinky'
]

matches = []
time_window = timedelta(milliseconds=200)

for _, g in gest.iterrows():
    gt = g["ParsedTime"]
    raw_x = g["GX"] 
    raw_y = g["GY"]
    if raw_x is None or raw_y is None:
        matches.append("Unknown")
        continue

    gx = raw_x * SCALE_X
    gy = raw_y * SCALE_Y

    window = fing[abs(fing["ParsedTime"] - gt) <= time_window]
    if window.empty:
        matches.append("Unknown")
        continue

    nearest = window.iloc[(window["ParsedTime"] - gt).abs().argmin()]

    best, best_d = "Unknown", float("inf")
    for col in finger_cols:
        coord = parse_finger_coord(nearest[col])
        if not coord:
            continue
        x, y = coord
        d = math.hypot(x - gx, y - gy)
        if d < best_d:
            best_d, best = d, col.replace("_"," ")
    matches.append(best)

gest["Finger"] = matches

out = gest[["Time","Type","X","Y","Keys","Finger"]]
now = datetime.now()
default_name = f"gesture_finger_match_{now.month:02d}-{now.day:02d}-{now.year}__{now.hour:02d}-{now.minute:02d}-{now.second:02d}.csv"

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
