import pandas as pd
from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import os

def parse_time(t):
    return datetime.strptime(t, "%H:%M:%S.%f")

def clean_val(val):
    val = val.strip("[]").replace("'", "").strip()
    return f"[{val}]"

def find_closest_finger(gesture_time, finger_df, window=200):
    min_time_diff = timedelta(milliseconds=window)
    closest_row = None

    for _, row in finger_df.iterrows():
        finger_time = parse_time(row["Timestamp"])
        time_diff = abs(gesture_time - finger_time)
        if time_diff <= min_time_diff:
            min_time_diff = time_diff
            closest_row = row

    return closest_row["Finger"] if closest_row is not None else "Unknown"

root = Tk()
root.withdraw()

gesture_path = filedialog.askopenfilename(title="Select Gesture Log CSV", filetypes=[("CSV Files", "*.csv")])
finger_path = filedialog.askopenfilename(title="Select Finger Log CSV", filetypes=[("CSV Files", "*.csv")])

gesture_df = pd.read_csv(gesture_path)
finger_df = pd.read_csv(finger_path)

gesture_df["ParsedTime"] = gesture_df["Time"].apply(parse_time)

gesture_df["X"] = gesture_df["X"].apply(clean_val)
gesture_df["Y"] = gesture_df["Y"].apply(clean_val)
gesture_df["Keys"] = gesture_df["Keys"].apply(clean_val)

fingers = []
for _, gesture_row in gesture_df.iterrows():
    g_time = gesture_row["ParsedTime"]
    matched_finger = find_closest_finger(g_time, finger_df)
    fingers.append(matched_finger)

gesture_df["Finger"] = fingers

result_df = gesture_df[["Time", "Type", "X", "Y", "Keys", "Finger"]]

save_path = filedialog.asksaveasfilename(
    title="Save Result CSV",
    defaultextension=".csv",
    filetypes=[("CSV Files", "*.csv")],
    initialfile="gesture_finger_match.csv"
)

result_df.to_csv(save_path, index=False)
print(f"Result saved to: {save_path}")
