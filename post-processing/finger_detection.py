import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np
import csv
from tkinter import Tk, filedialog
import os, tempfile, shutil

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

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

# ---------- helpers for lightweight video recording ----------
def make_temp_writer(start_stamp, fps, size):
    """
    Try several small codecs. Returns (writer, tmp_path, ext, codec) or (None, None, None, None).
    """
    tries = [
        ("avc1", ".mp4"),  # H.264 via FFMPEG (best size)
        ("H264", ".mp4"),
        ("X264", ".mp4"),
        ("mp4v", ".mp4"),  # MPEG-4 part 2 (okay size)
        ("XVID", ".avi"),  # larger, but widely supported
        ("MJPG", ".avi"),  # much larger, last resort
    ]
    for codec, ext in tries:
        tmp_path = os.path.join(tempfile.gettempdir(), f"ipad_warp_{start_stamp}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, size)
        if writer.isOpened():
            return writer, tmp_path, ext, codec
        else:
            writer.release()
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
    return None, None, None, None

clicked = []
def click_pts(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x,y))

list_ports()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cv2.namedWindow("Original Feed")
cv2.setMouseCallback("Original Feed", click_pts)
print("Click the FOUR corners of your screen area (e.g. iPad), from top left clockwise...")

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
# s = pts.sum(axis=1); d = np.diff(pts, axis=1)
ordered = np.zeros((4,2), dtype=np.float32)
# ordered[0] = pts[np.argmin(s)]
# ordered[2] = pts[np.argmax(s)]
# ordered[1] = pts[np.argmin(d)]
# ordered[3] = pts[np.argmax(d)]
ordered[0] = pts[0]
ordered[2] = pts[2]
ordered[1] = pts[1]
ordered[3] = pts[3]

src_pts = ordered
dst_pts = np.array([
    [0,0],
    [WARPED_SIZE[0],0],
    [WARPED_SIZE[0],WARPED_SIZE[1]],
    [0,WARPED_SIZE[1]]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("Tracking started—press 'q' to quit.")

# ---------- set up lightweight recording of the warped window ----------
# FPS can be 0 on some cameras; use a sane default if so.
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or fps > 120:  # guard weird values
    fps = 30.0

start_dt = datetime.now()
start_stamp = f"{start_dt.month:02d}-{start_dt.day:02d}-{start_dt.year}__{start_dt.hour:02d}-{start_dt.minute:02d}-{start_dt.second:02d}"

video_writer, tmp_video_path, used_ext, used_codec = make_temp_writer(start_stamp, fps, WARPED_SIZE)
if video_writer:
    print(f"Recording warped window to temporary file (codec {used_codec})…")
else:
    print("Video recording not available on this OpenCV build (no suitable codec). Continuing without recording.")

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

    # ---- write warped window to video (compressed) ----
    if video_writer:
        video_writer.write(warped)

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

# make sure writer is closed before moving the file
if video_writer:
    video_writer.release()

cv2.destroyAllWindows()

# ---------- save CSV ----------
now_dt = datetime.now()
default_csv = (
    f"finger_log_{now_dt.month:02d}-{now_dt.day:02d}-"
    f"{now_dt.year}__{now_dt.hour:02d}-"
    f"{now_dt.minute:02d}-{now_dt.second:02d}.csv"
)
save_path = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files","*.csv")],
    initialfile=default_csv,
    title="Save finger-log CSV"
)
if save_path:
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Timestamp"] + ALL_FINGERS)
        w.writerows(log_rows)
    print(f"Saved CSV to {save_path}")
else:
    print("No file chosen—no CSV written.")

# ---------- ask where to save the video ----------
if video_writer and tmp_video_path and os.path.exists(tmp_video_path):
    default_video = f"ipad_warp_{start_stamp}{used_ext}"
    vid_save_path = filedialog.asksaveasfilename(
        defaultextension=used_ext,
        filetypes=[("MP4 video","*.mp4"), ("AVI video","*.avi"), ("All files","*.*")],
        initialfile=default_video,
        title="Save warped-window video"
    )
    if vid_save_path:
        try:
            shutil.move(tmp_video_path, vid_save_path)
            print(f"Saved video to {vid_save_path}")
        except Exception as e:
            print(f"Could not move video to chosen location: {e}\nTemporary file left at: {tmp_video_path}")
    else:
        print(f"No video location chosen. Temporary file left at: {tmp_video_path}")
