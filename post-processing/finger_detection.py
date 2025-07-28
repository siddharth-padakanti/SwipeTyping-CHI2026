import cv2
import mediapipe as mp
import time
from datetime import datetime
import math
from collections import defaultdict
import numpy as np
import csv
from tkinter import Tk, filedialog

# Suppress the root window from showing
root = Tk()
root.withdraw()

# Store logs in memory during runtime
log_rows = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = {
    4: 'Thumb',
    8: 'Index',
    12: 'Middle',
    16: 'Ring',
    20: 'Pinky'
}

FINGER_JOINTS = {
    'Thumb':  [1, 2, 3, 4],
    'Index':  [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring':   [13, 14, 15, 16],
    'Pinky':  [17, 18, 19, 20]
}

FINGER_COLORS = {
    'Left_Thumb':    (0, 0, 255),      # Red
    'Left_Index':    (0, 255, 0),      # Green
    'Left_Middle':   (255, 0, 0),      # Blue
    'Left_Ring':     (0, 255, 255),    # Yellow
    'Left_Pinky':    (255, 0, 255),    # Magenta

    'Right_Thumb':   (128, 0, 255),    # Purple
    'Right_Index':   (0, 165, 255),    # Orange
    'Right_Middle':  (255, 255, 0),    # Cyan
    'Right_Ring':    (203, 192, 255),  # Pink-ish
    'Right_Pinky':   (0, 128, 255),    # Sky Blue
}

prev_positions = {}
movement_totals = defaultdict(float)
last_log_time = time.time()

clicked_points = []
warped_size = (640, 480)

def click_points(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Select Area (click 4 points)")
cv2.setMouseCallback("Select Area (click 4 points)", click_points)

print("Click 4 points on the screen to select the tracking area...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for pt in clicked_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("Select Area (click 4 points)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(clicked_points) == 4:
        break

cv2.destroyWindow("Select Area (click 4 points)")

if len(clicked_points) != 4:
    print("You didn't click 4 points. Exiting.")
    cap.release()
    exit()

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]     
    ordered[2] = pts[np.argmax(s)]      
    ordered[1] = pts[np.argmin(diff)]   
    ordered[3] = pts[np.argmax(diff)]    
    return ordered

ordered_pts = order_points(clicked_points)
src_pts = np.array(ordered_pts, dtype=np.float32)
dst_pts = np.array([
    [0, 0],
    [warped_size[0], 0],
    [warped_size[0], warped_size[1]],
    [0, warped_size[1]]
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("Tracking started in warped view...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    warped = cv2.warpPerspective(frame, matrix, warped_size)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    results = hands.process(warped_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            raw_label = handedness.classification[0].label  # 'Left' or 'Right'
            label = 'Right' if raw_label == 'Left' else 'Left'

            for idx, lm in enumerate(hand_landmarks.landmark):
                if idx in FINGER_TIPS:
                    h, w, _ = warped.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    finger_name = FINGER_TIPS[idx]
                    key = f"{label}_{finger_name}"

                    if key in prev_positions:
                        px, py = prev_positions[key]
                        dist = math.hypot(cx - px, cy - py)
                        movement_totals[key] += dist

                    prev_positions[key] = (cx, cy)

            mp_draw.draw_landmarks(warped, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for finger_name, indices in FINGER_JOINTS.items():
                key = f"{label}_{finger_name}"
                color = FINGER_COLORS.get(key, (255, 255, 255))

                for idx in indices:
                    lm = hand_landmarks.landmark[idx]
                    h, w, _ = warped.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(warped, (cx, cy), 6, color, -1)

    now = time.time()
    if now - last_log_time >= 0.1:
        if movement_totals:
            max_moved_finger = max(movement_totals, key=movement_totals.get)
            label, finger = max_moved_finger.split('_')
            rel_coords = prev_positions[max_moved_finger]
            current_time = time.strftime("%H:%M:%S", time.localtime(now))
            milliseconds = int((now % 1) * 1000)
            timestamp = f"{current_time}.{milliseconds:03d}"
            log_rows.append([timestamp, label, finger, [rel_coords[0]], [rel_coords[1]]])
        movement_totals.clear()
        last_log_time = now

    cv2.imshow("Warped Tracking Area", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ask user where to save after tracking ends
now = datetime.now()
default_filename = f"finger_log_{now.month:02d}-{now.day:02d}-{now.year}__{now.hour:02d}-{now.minute:02d}-{now.second:02d}.csv"

root = Tk()
root.withdraw()

save_path = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    initialfile=default_filename,
    title="Save tracking log as..."
)

if save_path:
    with open(save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp", "Hand", "Finger", "X", "Y"])
        csv_writer.writerows(log_rows)
    print(f"Tracking data saved to: {save_path}")
else:
    print("No file selected. Tracking data was not saved.")

csv_file.close()
print(f"Tracking data saved to: {save_path}")

