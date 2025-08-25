
"""
CSV + Video Sync & Editor (Pure Python, Tkinter)
------------------------------------------------
What it does:
- Lets you choose a video file and a CSV log file.
- Displays video (left) and CSV table (right) side-by-side.
- Parses a start time from the *video filename* (e.g., "..._08-24-2025__20-15-58.mp4"),
  then syncs the CSV rows via their "Timestamp"/"Time" column.
- Scrub with the timeline slider; the video frame updates and the nearest CSV row highlights.
- Click a CSV row to jump the timeline to that moment.
- Double-click a CSV cell to edit it in-place.
- Save the edited CSV via a file dialog.
- NEW: Play/Pause button and Spacebar shortcut to play/pause; ±0.5s nudge buttons.

Dependencies (install if needed):
    pip install opencv-python pandas pillow

Run:
    python csv_video_sync_tool.py

Notes:
- Works on Windows/macOS/Linux with Python 3.9+.
- Requires a display (it's a desktop app).

Author: ChatGPT (for Sid)
"""
import os
import re
import sys
import math
import threading
from datetime import datetime, date, time, timedelta
from typing import Optional, List

# Third-party
try:
    import cv2
except Exception as e:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise
try:
    import pandas as pd
except Exception as e:
    print("pandas is required. Install with: pip install pandas")
    raise
try:
    from PIL import Image, ImageTk
except Exception as e:
    print("Pillow is required. Install with: pip install pillow")
    raise

# Tkinter / ttk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def parse_video_start_from_name(path: str) -> Optional[datetime]:
    """
    Try to parse a start datetime from a video filename.
    Supported patterns (anywhere in the filename):
        MM-DD-YYYY__HH-MM-SS
        MM-DD-YYYY_HH-MM-SS
        YYYY-MM-DD__HH-MM-SS
        YYYY-MM-DD_HH-MM-SS

    Examples:
        "ipad_tap_08-24-2025__20-15-58.mp4"
        "2025-08-24_20-15-58_cam1.mov"
    """
    fname = os.path.basename(path)

    patterns = [
        r'(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})[_-]{1,2}(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})',
        r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[_-]{1,2}(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})',
    ]

    for pat in patterns:
        m = re.search(pat, fname)
        if m:
            parts = m.groupdict()
            try:
                y = int(parts["year"])
                mo = int(parts["month"])
                d = int(parts["day"])
                hh = int(parts["hour"])
                mm = int(parts["minute"])
                ss = int(parts["second"])
                return datetime(y, mo, d, hh, mm, ss)
            except Exception:
                pass
    return None


def parse_timestamp_field(s: str) -> Optional[datetime]:
    """
    Parse a CSV timestamp string into a datetime.
    Supports:
        - "HH:MM:SS[.fff]" (we will combine with *today's* date at runtime; caller can adjust to desired date)
        - "YYYY-MM-DD HH:MM:SS[.fff]"
        - "MM/DD/YYYY HH:MM:SS[.fff]"
        - "YYYY-MM-DDTHH:MM:SS[.fff]"
    Returns None on failure.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()

    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%H:%M:%S.%f",
        "%H:%M:%S",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt
        except ValueError:
            continue
    return None


class EditableTreeview(ttk.Treeview):
    """
    ttk.Treeview with double-click cell editing.
    Updates are propagated via a callback.
    """
    def __init__(self, master, columns: List[str], on_cell_edited, **kwargs):
        super().__init__(master, columns=columns, show="headings", **kwargs)
        self.columns_list = columns
        self.on_cell_edited = on_cell_edited
        self._editor = None
        self._editor_col = None
        self._editor_item = None

        # Headers
        for c in columns:
            self.heading(c, text=c, anchor="w")
            self.column(c, anchor="w", width=120, stretch=True)

        self.bind("<Double-1>", self._begin_edit)
        self.bind("<Escape>", self._cancel_edit)

    def _begin_edit(self, event):
        if self._editor:
            self._finish_edit(commit=True)

        region = self.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self.identify_row(event.y)
        col_id = self.identify_column(event.x)  # e.g. '#1'
        if not row_id or not col_id:
            return

        col_index = int(col_id[1:]) - 1
        if col_index < 0 or col_index >= len(self.columns_list):
            return

        x, y, w, h = self.bbox(row_id, col_id)
        value = self.set(row_id, self.columns_list[col_index])

        self._editor_item = row_id
        self._editor_col = self.columns_list[col_index]
        self._editor = tk.Entry(self, borderwidth=1)
        self._editor.insert(0, value)
        self._editor.select_range(0, tk.END)
        self._editor.focus_set()
        self._editor.place(x=x, y=y, width=w, height=h)
        self._editor.bind("<Return>", lambda e: self._finish_edit(commit=True))
        self._editor.bind("<FocusOut>", lambda e: self._finish_edit(commit=True))

    def _cancel_edit(self, event=None):
        if self._editor:
            self._editor.destroy()
            self._editor = None
            self._editor_col = None
            self._editor_item = None

    def _finish_edit(self, commit: bool):
        if not self._editor:
            return
        new_value = self._editor.get()
        item = self._editor_item
        col = self._editor_col

        self._editor.destroy()
        self._editor = None

        if commit and item and col:
            old = self.set(item, col)
            if new_value != old:
                self.set(item, col, new_value)
                try:
                    idx = int(self.index(item))
                except Exception:
                    idx = None
                if idx is not None and self.on_cell_edited:
                    self.on_cell_edited(idx, col, new_value)

        self._editor_col = None
        self._editor_item = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV + Video Sync & Editor (Python/Tkinter)")
        self.geometry("1200x760")

        # State
        self.video_path: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.df_cols: List[str] = []
        self.abs_times: Optional[List[datetime]] = None  # normalized absolute datetimes per row
        self.video_start_dt: Optional[datetime] = None
        self.time_offset_sec: float = 0.0  # offset applied so that "slider time = 0" aligns to video_start_dt

        # Video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_duration_sec: float = 0.0
        self.video_fps: float = 0.0

        # Playback state
        self.playing: bool = False
        self.play_interval_ms: int = 33  # ~30fps default

        self.highlight_tag = "highlight_tag"

        self._build_ui()

        # Keyboard shortcuts
        self.bind("<space>", lambda e: self.toggle_play())

        # Graceful close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Video...", command=self.open_video)
        filemenu.add_command(label="Open CSV...", command=self.open_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Save CSV As...", command=self.save_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        syncmenu = tk.Menu(menubar, tearoff=0)
        syncmenu.add_command(label="Use Video Filename Start", command=self.use_video_filename_start)
        syncmenu.add_command(label="Use CSV's First Timestamp as Start", command=self.use_csv_first_timestamp_start)
        syncmenu.add_command(label="Set Manual Offset...", command=self.set_manual_offset_dialog)
        menubar.add_cascade(label="Sync", menu=syncmenu)

        self.config(menu=menubar)

        # Paned layout
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: Video
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        self.video_canvas = tk.Canvas(left_frame, bg="#111", width=800, height=450, highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Right: CSV table
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        self.table_container = ttk.Frame(right_frame)
        self.table_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.table: Optional[EditableTreeview] = None  # created after CSV loaded
        self.table_scroll_y = ttk.Scrollbar(self.table_container, orient="vertical")
        self.table_scroll_x = ttk.Scrollbar(self.table_container, orient="horizontal")

        # Bottom: Controls + Timeline
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)

        # Play/Pause and nudge controls
        self.play_btn = ttk.Button(bottom, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=(0,6))

        ttk.Button(bottom, text="-0.5s", command=lambda: self.nudge_time(-0.5)).pack(side=tk.LEFT)
        ttk.Button(bottom, text="+0.5s", command=lambda: self.nudge_time(+0.5)).pack(side=tk.LEFT, padx=(6,12))

        self.time_label = ttk.Label(bottom, text="00:00.000 / 00:00.000")
        self.time_label.pack(side=tk.LEFT, padx=(0, 12))

        self.timeline = ttk.Scale(bottom, from_=0.0, to=1.0, value=0.0, command=self.on_slider_move)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.jump_entry = ttk.Entry(bottom, width=12)
        self.jump_entry.insert(0, "0:00.000")
        self.jump_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="Jump", command=self.jump_to_time).pack(side=tk.LEFT)

        # Status bar
        self.status = tk.StringVar(value="Open a video and a CSV to begin.")
        statusbar = ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w")
        statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- File handling ----------
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")]
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str):
        # Stop playback if running
        if self.playing:
            self.toggle_play(force=False)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(path)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Video", "Failed to open video.")
            return

        self.video_path = path
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.video_duration_sec = float(frame_count) / float(self.video_fps) if self.video_fps > 0 else 0.0

        # Playback timer interval
        self.play_interval_ms = max(10, int(1000.0 / (self.video_fps if self.video_fps > 0 else 30.0)))

        # Parse start time from filename (if possible)
        start_dt = parse_video_start_from_name(path)
        if start_dt:
            self.video_start_dt = start_dt
            self.status.set(f"Video loaded. Start parsed from filename: {start_dt}")
        else:
            self.video_start_dt = None
            self.status.set("Video loaded. Could not parse start time from filename (use Sync menu).")

        # Update timeline
        self.timeline.configure(from_=0.0, to=max(0.001, self.video_duration_sec))
        self.update_time_label(0.0, self.video_duration_sec)

        # Show first frame
        self.show_frame_at_time(0.0)

    def open_csv(self):
        path = filedialog.askopenfilename(
            title="Choose CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self._load_csv(path)

    def _load_csv(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("CSV", f"Failed to read CSV: {e}")
            return

        self.csv_path = path
        self.df = df
        self.df_cols = list(df.columns)

        # Ensure we have a timestamp-like column
        ts_col = None
        for candidate in ["Timestamp", "Time", "timestamp", "time"]:
            if candidate in self.df_cols:
                ts_col = candidate
                break
        if ts_col is None:
            messagebox.showwarning("CSV", "No 'Timestamp' or 'Time' column found. Sync will be disabled.")
            self.abs_times = None
        else:
            # Parse timestamps to absolute datetimes
            self.abs_times = []
            inferred_date = None
            if self.video_start_dt:
                # Use the date from the parsed start dt if CSV has only times
                inferred_date = self.video_start_dt.date()

            for i, s in enumerate(self.df[ts_col].astype(str).tolist()):
                dt = parse_timestamp_field(s)
                if dt is None:
                    self.abs_times.append(None)
                    continue
                # If parsed only time-of-day (year=1900), combine with inferred date or today
                if dt.year == 1900 and dt.month == 1 and dt.day == 1:
                    if inferred_date is None:
                        inferred_date = date.today()
                    dt = datetime.combine(inferred_date, dt.time())
                self.abs_times.append(dt)

        # Build / refresh table UI
        self._build_table()

        self.status.set(f"CSV loaded with {len(self.df)} rows.")

        # If video has no start time yet, try to use CSV first timestamp
        if (self.video_start_dt is None) and self.abs_times:
            first_valid = next((t for t in self.abs_times if t is not None), None)
            if first_valid:
                self.video_start_dt = first_valid
                self.status.set(f"CSV loaded. Using CSV first timestamp as video start: {first_valid}")

    def _build_table(self):
        # Clear previous
        for w in self.table_container.winfo_children():
            w.destroy()

        self.table_scroll_y = ttk.Scrollbar(self.table_container, orient="vertical")
        self.table_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.table_scroll_x = ttk.Scrollbar(self.table_container, orient="horizontal")
        self.table_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.table = EditableTreeview(
            self.table_container,
            columns=self.df_cols,
            on_cell_edited=self.on_cell_edited,
            yscrollcommand=self.table_scroll_y.set,
            xscrollcommand=self.table_scroll_x.set,
            selectmode="browse",
            height=25,
        )
        self.table.pack(fill=tk.BOTH, expand=True)

        self.table_scroll_y.config(command=self.table.yview)
        self.table_scroll_x.config(command=self.table.xview)

        # Insert rows
        if self.df is not None:
            for _, row in self.df.iterrows():
                vals = [str(row[c]) if c in self.df_cols else "" for c in self.df_cols]
                self.table.insert("", tk.END, values=vals)

        # Style for highlight
        self.table.tag_configure(self.highlight_tag, background="#fdf5d8")

        # Select → jump timeline
        self.table.bind("<<TreeviewSelect>>", self.on_table_select)

    # ---------- Sync helpers ----------
    def use_video_filename_start(self):
        if not self.video_path:
            messagebox.showinfo("Sync", "Open a video first.")
            return
        dt = parse_video_start_from_name(self.video_path)
        if not dt:
            messagebox.showwarning("Sync", "Could not parse a start time from the video filename.")
            return
        self.video_start_dt = dt
        self.status.set(f"Start set from video filename: {dt}")
        # Refresh highlight for current position
        self.on_slider_move(self.timeline.get())

    def use_csv_first_timestamp_start(self):
        if not self.abs_times:
            messagebox.showinfo("Sync", "No CSV timestamps available.")
            return
        first_valid = next((t for t in self.abs_times if t is not None), None)
        if not first_valid:
            messagebox.showinfo("Sync", "CSV has no parsable timestamps.")
            return
        self.video_start_dt = first_valid
        self.status.set(f"Start set from CSV first timestamp: {first_valid}")
        self.on_slider_move(self.timeline.get())

    def set_manual_offset_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Set Manual Offset (seconds)")
        ttk.Label(dlg, text="Offset seconds (positive = shift CSV later):").pack(padx=10, pady=10)
        e = ttk.Entry(dlg)
        e.insert(0, f"{self.time_offset_sec:.3f}")
        e.pack(padx=10, pady=(0,10))
        def ok():
            try:
                self.time_offset_sec = float(e.get().strip())
                self.status.set(f"Manual offset applied: {self.time_offset_sec:.3f} s")
                self.on_slider_move(self.timeline.get())
                dlg.destroy()
            except Exception:
                messagebox.showerror("Offset", "Enter a valid number.")
        ttk.Button(dlg, text="OK", command=ok).pack(pady=(0,10))
        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)

    def update_time_label(self, t_cur: float, t_total: float):
        def fmt(secs: float) -> str:
            if secs < 0: secs = 0
            msec = int(round((secs - int(secs)) * 1000))
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:d}:{m:02d}:{s:02d}.{msec:03d}"
            else:
                return f"{m:d}:{s:02d}.{msec:03d}"
        self.time_label.config(text=f"{fmt(t_cur)} / {fmt(t_total)}")

    def absolute_time_for_slider(self, slider_sec: float) -> Optional[datetime]:
        if self.video_start_dt is None:
            return None
        return self.video_start_dt + timedelta(seconds=slider_sec + self.time_offset_sec)

    def find_nearest_row_by_abs_time(self, target: datetime) -> Optional[int]:
        if not self.abs_times:
            return None
        best_idx = None
        best_delta = None
        for i, t in enumerate(self.abs_times):
            if t is None: 
                continue
            delta = abs((t - target).total_seconds())
            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_idx = i
        return best_idx

    def highlight_row(self, idx: Optional[int]):
        # Clear all tags first (cheap way: rebuild tags)
        for item in self.table.get_children():
            self.table.item(item, tags=())
        if idx is None:
            return
        # Get item id by index
        try:
            item_id = self.table.get_children()[idx]
        except Exception:
            return
        self.table.item(item_id, tags=(self.highlight_tag,))

        # Ensure visible
        self.table.see(item_id)

    # ---------- Playback ----------
    def toggle_play(self, force: Optional[bool] = None):
        if force is not None:
            self.playing = force
        else:
            self.playing = not self.playing

        self.play_btn.config(text="Pause" if self.playing else "Play")

        if self.playing:
            self._play_loop()

    def _play_loop(self):
        if not self.playing:
            return
        if not self.cap:
            self.toggle_play(force=False)
            return
        cur = float(self.timeline.get())
        step = 1.0 / (self.video_fps if self.video_fps > 0 else 30.0)
        nxt = cur + step
        if nxt >= self.video_duration_sec:
            # Stop at end
            self.timeline.set(self.video_duration_sec)
            self.on_slider_move(self.video_duration_sec)
            self.toggle_play(force=False)
            return
        self.timeline.set(nxt)
        self.on_slider_move(nxt)
        self.after(self.play_interval_ms, self._play_loop)

    def nudge_time(self, delta: float):
        t = float(self.timeline.get())
        t = max(0.0, min(self.video_duration_sec, t + delta))
        self.timeline.set(t)
        self.on_slider_move(t)

    # ---------- Events ----------
    def on_slider_move(self, value):
        try:
            t = float(value)
        except Exception:
            return
        self.update_time_label(t, self.video_duration_sec)
        self.show_frame_at_time(t)

        # Highlight CSV row for current absolute time
        abs_t = self.absolute_time_for_slider(t)
        if abs_t is not None:
            idx = self.find_nearest_row_by_abs_time(abs_t)
            self.highlight_row(idx)

    def show_frame_at_time(self, sec: float):
        if not self.cap:
            return
        # Seek to msec
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0, sec) * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return

        # Convert BGR→RGB and fit to canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Fit to canvas while preserving aspect
        canvas_w = max(1, self.video_canvas.winfo_width())
        canvas_h = max(1, self.video_canvas.winfo_height())

        img_w, img_h = img.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        img_resized = img.resize(new_size, Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(img_resized)
        self.video_canvas.delete("all")
        # Center
        x = (canvas_w - new_size[0]) // 2
        y = (canvas_h - new_size[1]) // 2
        self.video_canvas.create_image(x, y, image=self._tk_img, anchor="nw")

    def on_table_select(self, event):
        # Jump timeline to selected row's time
        if not self.table or not self.abs_times:
            return
        sel = self.table.selection()
        if not sel:
            return
        item_id = sel[0]
        try:
            idx = self.table.index(item_id)
        except Exception:
            return
        abs_t = self.abs_times[idx] if idx < len(self.abs_times) else None
        if abs_t and self.video_start_dt:
            sec = (abs_t - self.video_start_dt).total_seconds() - self.time_offset_sec
            sec = max(0.0, min(self.video_duration_sec, sec))
            self.timeline.set(sec)
            self.on_slider_move(sec)

    def on_cell_edited(self, row_idx: int, col_name: str, new_value: str):
        # Update pandas DataFrame
        if self.df is None:
            return
        try:
            self.df.at[row_idx, col_name] = new_value
            # If editing Timestamp column, update abs_times
            if col_name.lower() in ("timestamp", "time"):
                dt = parse_timestamp_field(new_value)
                if dt is not None and dt.year == 1900 and dt.month == 1 and dt.day == 1:
                    # combine with video_start date if available
                    base_date = self.video_start_dt.date() if self.video_start_dt else date.today()
                    dt = datetime.combine(base_date, dt.time())
                if self.abs_times is None:
                    self.abs_times = [None] * len(self.df)
                if row_idx < len(self.abs_times):
                    self.abs_times[row_idx] = dt
            self.status.set(f"Edited row {row_idx+1}, column '{col_name}'.")
        except Exception as e:
            messagebox.showerror("Edit", f"Failed to apply edit: {e}")

    def save_csv(self):
        if self.df is None:
            messagebox.showinfo("Save", "Nothing to save (no CSV loaded).")
            return
        path = filedialog.asksaveasfilename(
            title="Save edited CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            self.status.set(f"Saved: {path}")
            messagebox.showinfo("Save", f"CSV saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save CSV: {e}")

    def jump_to_time(self):
        s = self.jump_entry.get().strip()
        # Accept formats like "m:ss.mmm" or "h:mm:ss.mmm" or a plain float seconds
        sec = None
        # Try float
        try:
            sec = float(s)
        except Exception:
            pass
        if sec is None:
            # Try h:mm:ss.mmm
            m = re.match(r'(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$', s)
            if m:
                h = int(m.group(1)) if m.group(1) else 0
                m_ = int(m.group(2))
                s_ = int(m.group(3))
                ms = int(m.group(4)) if m.group(4) else 0
                sec = h*3600 + m_*60 + s_ + ms/1000.0
        if sec is None:
            messagebox.showerror("Jump", "Enter seconds (float) or time like h:mm:ss.mmm")
            return
        sec = max(0.0, min(self.video_duration_sec, sec))
        self.timeline.set(sec)
        self.on_slider_move(sec)

    def _on_close(self):
        # Stop playback and release resources
        self.playing = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
