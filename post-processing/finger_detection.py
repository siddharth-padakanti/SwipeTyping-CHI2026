import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np
import csv
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}
FINGER_JOINTS = {
    'Thumb':  [1,2,3,4],
    'Index':  [5,6,7,8],
    'Middle': [9,10,11,12],
    'Ring':   [13,14,15,16],
    'Pinky':  [17,18,19,20]
}
FINGER_COLORS = {
    'Left_Thumb':   (0,0,255),    'Left_Index':   (0,255,0),
    'Left_Middle':  (255,0,0),    'Left_Ring':    (0,255,255),
    'Left_Pinky':   (255,0,255),  'Right_Thumb':  (128,0,255),
    'Right_Index':  (0,165,255),  'Right_Middle': (255,255,0),
    'Right_Ring':   (203,192,255),'Right_Pinky':  (0,128,255)
}
ALL_FINGERS = [
    'Left_Pinky','Left_Ring','Left_Middle','Left_Index','Left_Thumb',
    'Right_Thumb','Right_Index','Right_Middle','Right_Ring','Right_Pinky'
]

prev_positions = {}
last_log_time = time.time()
log_rows = []
WARPED_SIZE = (640, 480)  # (width, height)

clicked = []
def click_pts(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x,y))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Original Feed")
cv2.setMouseCallback("Original Feed", click_pts)
print("Click the FOUR corners of your screen area (e.g. iPad)...")

while True:
    ret, frame = cap.read()
    if not ret: break
    for p in clicked:
        cv2.circle(frame, p, 5, (0,0,255), -1)
    cv2.imshow("Original Feed", frame)
    if cv2.waitKey(1)&0xFF == ord('q') or len(clicked)==4:
        break

if len(clicked) != 4:
    print("Need exactly 4 points. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

pts = np.array(clicked, dtype=np.float32)
s = pts.sum(axis=1); d = np.diff(pts, axis=1)
ordered = np.zeros((4,2), dtype=np.float32)
ordered[0] = pts[np.argmin(s)]
ordered[2] = pts[np.argmax(s)]
ordered[1] = pts[np.argmin(d)]
ordered[3] = pts[np.argmax(d)]

src_pts = ordered
dst_pts = np.array([
    [0,0],
    [WARPED_SIZE[0],0],
    [WARPED_SIZE[0],WARPED_SIZE[1]],
    [0,WARPED_SIZE[1]]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("Tracking started—press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    cv2.polylines(frame,
                  [src_pts.reshape(-1,1,2).astype(int)],
                  True,
                  (235,206,135),
                  2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    warped = cv2.warpPerspective(frame, M, WARPED_SIZE)

    if results.multi_hand_landmarks:
        h_o, w_o = frame.shape[:2]
        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                         results.multi_handedness or []):
            raw = handedness.classification[0].label  # 'Left' or 'Right'
            label = 'Right' if raw=='Left' else 'Left'

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            for tip_idx, tip_name in FINGER_TIPS.items():
                lm = hand_lms.landmark[tip_idx]
                ox, oy = int(lm.x * w_o), int(lm.y * h_o)
                pt = np.array([[[ox, oy]]], dtype=np.float32)
                wx, wy = cv2.perspectiveTransform(pt, M)[0][0]
                ux, uy = int(wx), int(wy)
                key = f"{label}_{tip_name}"
                if 0 <= ux < WARPED_SIZE[0] and 0 <= uy < WARPED_SIZE[1]:
                    prev_positions[key] = (ux, uy)
                    color = FINGER_COLORS.get(key, (255,255,255))
                    cv2.circle(warped, (ux, uy), 6, color, -1)
                else:
                    prev_positions.pop(key, None)

    cv2.imshow("Original Feed", frame)
    cv2.imshow("Warped Tracking Area", warped)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

    now = time.time()
    if now - last_log_time >= 0.1:
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        ms = int((now % 1)*1000)
        timestamp = f"{ts}.{ms:03d}"
        row = [timestamp] + [
            f"[{prev_positions[f][0]},{prev_positions[f][1]}]"
            if f in prev_positions else "-"
            for f in ALL_FINGERS
        ]
        log_rows.append(row)
        last_log_time = now

cap.release()
cv2.destroyAllWindows()


now_dt = datetime.now()
default = (
    f"finger_log_{now_dt.month:02d}-{now_dt.day:02d}-"
    f"{now_dt.year}__{now_dt.hour:02d}-"
    f"{now_dt.minute:02d}-{now_dt.second:02d}.csv"
)
save_path = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files","*.csv")],
    initialfile=default,
    title="Save finger-log CSV"
)
if save_path:
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Timestamp"] + ALL_FINGERS)
        w.writerows(log_rows)
    print(f"Saved to {save_path}")
else:
    print("No file chosen—no CSV written.")
