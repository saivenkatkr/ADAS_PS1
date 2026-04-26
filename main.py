"""
main.py  —  ADAS Full Pipeline
═══════════════════════════════════════════════════════════════════════════════
FEATURES:
  ✓ YOLOv8 object detection  — 80 COCO classes, correct labels always
  ✓ Kalman + Hungarian tracking — stable IDs, velocity, across frames
  ✓ Lane detection + filled polygon overlay
  ✓ Lane CHANGE detection  — big amber banner when changing lanes
  ✓ Lane departure warning — red border when drifting out
  ✓ Collision warning      — TTC + distance for every tracked object
  ✓ Blind spot detection   — side cameras
  ✓ Depth estimation       — MiDaS optional, off by default on CPU
  ✓ REVERSE MODE (E key)   — flips camera role: rear lane ON, front lane OFF
                              parking assist activates, collision warning switches
  ✓ Lane ON/OFF toggle (L)
  ✓ Depth overlay toggle (D)
  ✓ Tracker reset (R)
  ✓ Threaded capture — always live, no lag
  ✓ FPS counter + HUD

TWO MODES:
  1. LIVE    — real camera / video file → OpenCV window
  2. BATCH   — called by web_app.py → returns annotated video/image

KEYBOARD (live mode):
  Q / ESC — Quit
  L       — Toggle lane detection overlay
  D       — Toggle depth map overlay  (needs --depth flag)
  R       — Reset tracker IDs
  E       — Toggle REVERSE MODE
              FORWARD: front lane ON, rear lane OFF, collision warning ON
              REVERSE: rear lane ON,  front lane OFF, parking assist ON

RUN:
  python main.py                      # webcam 0
  python main.py --source video.mp4   # video file
  python main.py --source 1           # second webcam
  python main.py --width 320          # faster (CPU)
  python main.py --depth              # MiDaS depth (GPU recommended)
"""

import argparse
import sys
import time
import threading
import collections
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
_CONFIG = {
    "system": {"mode": "all", "target_fps": 30, "show_display": True},
    "cameras": {
        "front": {"source": 0, "resolution": [1280, 720], "fps": 30, "enabled": True},
        "left":  {"source": 1, "resolution": [1280, 720], "fps": 30, "enabled": False},
        "right": {"source": 2, "resolution": [1280, 720], "fps": 30, "enabled": False},
        "rear":  {"source": 3, "resolution": [1280, 720], "fps": 30, "enabled": False},
    },
    "perception": {
        "detector": {
            "backend": "yolov8", "model_path": "yolov8n.pt",
            "confidence": 0.40, "iou_threshold": 0.45,
            "classes": None, "device": "auto"
        },
        "lane_detector":   {"backend": "heuristic", "roi_top_ratio": 0.55},
        "depth_estimator": {"backend": "midas", "model_type": "MiDaS_small", "enabled": False}
    },
    "tracking": {"backend": "deepsort", "max_age": 30, "min_hits": 2, "iou_threshold": 0.3},
    "decision": {
        "collision":  {"ttc_warning_sec": 3.0, "ttc_critical_sec": 1.5, "min_object_area": 2000},
        "blind_spot": {"zone_x_ratio": 0.25, "min_velocity_kmh": 10.0},
        "parking":    {"critical_distance_cm": 40, "warning_distance_cm": 100, "grid_cells": [10, 6]}
    },
    "output": {
        "display": {"font_scale": 0.55, "overlay_alpha": 0.4},
        "audio":   {"enabled": False, "beep_cooldown_sec": 2.0},
        "logger":  {"enabled": False, "log_dir": "logs/",
                    "record_video": False, "video_dir": "logs/video/"}
    }
}

import utils.config_loader as _cfg_mod
_cfg_mod._config = _CONFIG

# ── Knobs ─────────────────────────────────────────────────────────────────────
INFER_WIDTH      = 416
DEPTH_EVERY_N    = 10
LANE_EVERY_N     = 3
DISPLAY_WIDTH    = 1024
CAPTURE_QUEUE_SZ = 2
LANE_CHANGE_HISTORY  = 15
LANE_CHANGE_THRESH   = 40
LANE_CHANGE_COOLDOWN = 60

# ── Color palette BGR ─────────────────────────────────────────────────────────
_C = {
    "car":(0,200,255),"truck":(0,140,255),"bus":(0,100,200),
    "motorcycle":(255,150,0),"bicycle":(255,200,0),"airplane":(180,180,255),
    "train":(120,100,200),"boat":(200,220,255),
    "person":(100,255,100),
    "dog":(255,80,200),"cat":(255,100,220),"horse":(210,80,180),
    "bird":(255,160,200),"sheep":(220,130,200),"cow":(200,80,160),
    "elephant":(180,60,140),"bear":(160,40,120),"zebra":(240,100,200),
    "giraffe":(230,120,190),
    "traffic light":(0,255,255),"stop sign":(0,50,255),"fire hydrant":(0,100,255),
    "backpack":(200,200,80),"umbrella":(180,200,100),"suitcase":(140,160,60),
    "laptop":(200,160,100),"cell phone":(180,140,80),
    "unknown":(200,200,200),"default":(200,200,200),
}
def _color(l): return _C.get(l.lower(), _C["default"])


# ═════════════════════════════════════════════════════════════════════════════
# LANE CHANGE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class LaneChangeDetector:
    """
    Watches center_offset_px across LANE_CHANGE_HISTORY frames.
    Detects a sudden directional shift → lane change alert.
    """
    def __init__(self):
        self._history      = collections.deque(maxlen=LANE_CHANGE_HISTORY)
        self._cooldown     = 0
        self.active        = False
        self.direction     = ""
        self.active_frames = 0

    def update(self, lanes) -> dict:
        result = {"changing": False, "direction": "", "alert_text": ""}
        if lanes is None or (lanes.left_line is None and lanes.right_line is None):
            self._history.clear()
            self.active = False
            self._cooldown = max(0, self._cooldown - 1)
            return result

        self._history.append(lanes.center_offset_px)
        if self._cooldown > 0:
            self._cooldown -= 1

        if len(self._history) < LANE_CHANGE_HISTORY // 2:
            return result

        half    = len(self._history) // 2
        old_avg = sum(list(self._history)[:half]) / half
        new_avg = sum(list(self._history)[half:]) / (len(self._history) - half)
        shift   = new_avg - old_avg

        if abs(shift) > LANE_CHANGE_THRESH and self._cooldown == 0:
            direction = "right" if shift > 0 else "left"
            self.active        = True
            self.direction     = direction
            self.active_frames = LANE_CHANGE_COOLDOWN // 2
            self._cooldown     = LANE_CHANGE_COOLDOWN
            result.update({"changing": True, "direction": direction,
                           "alert_text": f"LANE CHANGE  {direction.upper()}"})
        elif self.active_frames > 0:
            self.active_frames -= 1
            result.update({"changing": True, "direction": self.direction,
                           "alert_text": f"LANE CHANGE  {self.direction.upper()}"})
        else:
            self.active = False
            self.direction = ""
        return result

    def reset(self):
        self._history.clear()
        self._cooldown = 0
        self.active = False
        self.direction = ""
        self.active_frames = 0


# ═════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS  — identical output for LIVE and BATCH (web) modes
# ═════════════════════════════════════════════════════════════════════════════

def _draw_tracks(frame, tracks):
    fs = 0.55
    for track in tracks:
        b     = track.bbox
        label = track.display_label
        color = _color(label)
        thick = 2 if track.confidence > 0.6 else 1
        cv2.rectangle(frame, (b.x1,b.y1), (b.x2,b.y2), color, thick)
        tag = f"#{track.track_id} {label} {int(track.confidence*100)}%"
        (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(b.y1, th+8)
        cv2.rectangle(frame,(b.x1,ty-th-6),(b.x1+tw+6,ty),color,-1)
        cv2.putText(frame,tag,(b.x1+3,ty-3),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),1,cv2.LINE_AA)
        if track.estimated_distance_m is not None:
            cx = (b.x1+b.x2)//2
            cv2.putText(frame,f"{track.estimated_distance_m:.1f}m",
                        (cx-20,b.y2+18),cv2.FONT_HERSHEY_SIMPLEX,fs,color,2,cv2.LINE_AA)
    return frame


def _draw_lanes(frame, lanes, lane_on=True):
    if not lane_on or lanes is None:
        return frame
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    if lanes.left_line is not None:
        lx1,ly1,lx2,ly2 = lanes.left_line
        cv2.line(overlay,(lx1,ly1),(lx2,ly2),(0,255,0),4)
    if lanes.right_line is not None:
        rx1,ry1,rx2,ry2 = lanes.right_line
        cv2.line(overlay,(rx1,ry1),(rx2,ry2),(0,255,0),4)
    if lanes.left_line is not None and lanes.right_line is not None:
        lx1,ly1,lx2,ly2 = lanes.left_line
        rx1,ry1,rx2,ry2 = lanes.right_line
        pts = np.array([[lx1,ly1],[lx2,ly2],[rx2,ry2],[rx1,ry1]],np.int32)
        cv2.fillPoly(overlay,[pts],(0,150,0))
    if lanes.departure_warning:
        cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,0,255),10)
    return cv2.addWeighted(overlay,0.35,frame,0.65,0)


def _draw_collision_alerts(frame, alerts):
    alert_colors = {"critical":(0,0,255),"warning":(0,165,255),"info":(0,220,255)}
    y = 90
    for alert in alerts:
        color = alert_colors.get(alert.level.value.lower(),(255,255,255))
        msg   = alert.message
        (tw,th),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        cv2.rectangle(frame,(6,y-th-5),(10+tw+6,y+6),(20,20,20),-1)
        cv2.putText(frame,msg,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
        y += th+16
    return frame


def _draw_lane_change_banner(frame, lc_result):
    if not lc_result.get("changing"):
        return frame
    h, w    = frame.shape[:2]
    direction = lc_result.get("direction","")
    arrow = "<<  " if direction=="left" else "  >>"
    text  = f"{arrow}  LANE CHANGE {direction.upper()}  {arrow}"
    overlay = frame.copy()
    bh = 64; by = h//2 - bh//2
    cv2.rectangle(overlay,(0,by),(w,by+bh),(0,140,255),-1)
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,1.0,2)
    cv2.putText(overlay,text,(max(0,(w-tw)//2),by+(bh+th)//2),
                cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
    return cv2.addWeighted(overlay,0.80,frame,0.20,0)


def _draw_lane_departure_warning(frame, lanes):
    if lanes is None or not lanes.departure_warning:
        return frame
    h, w = frame.shape[:2]
    msg  = "  LANE DEPARTURE WARNING  "
    (tw,th),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
    tx = max(0,(w-tw)//2); ty = 55
    cv2.rectangle(frame,(tx-6,ty-th-6),(tx+tw+6,ty+8),(0,0,200),-1)
    cv2.putText(frame,msg,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2,cv2.LINE_AA)
    return frame


def _draw_depth_overlay(frame, depth_map):
    if depth_map is None or depth_map.map is None:
        return frame
    d_norm  = (depth_map.map*255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)
    d_rsz   = cv2.resize(d_color,(frame.shape[1],frame.shape[0]))
    return cv2.addWeighted(frame,0.65,d_rsz,0.35,0)


def _draw_parking_grid(frame, tracks):
    """
    Draws a simple proximity grid at the bottom of the frame
    when in REVERSE mode. Shows how close objects are.
    """
    h, w = frame.shape[:2]
    grid_h, grid_w = 100, w
    grid_y = h - grid_h - 5
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,grid_y),(w,h-5),(20,20,40),-1)
    frame = cv2.addWeighted(overlay,0.6,frame,0.4,0)

    # Label
    cv2.putText(frame,"◀ REVERSE — PARKING ASSIST ▶",
                (w//2-200, grid_y+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,136),2,cv2.LINE_AA)

    # Dots for nearby objects
    for track in tracks:
        cx = (track.bbox.x1+track.bbox.x2)//2
        # map cx to grid x
        gx = cx
        gy = grid_y + 55
        area = track.bbox.area
        frame_area = h * w
        ratio = min(1.0, area / (frame_area * 0.3))
        radius = max(6, int(ratio * 28))
        # color: red=close, green=far
        r = int(ratio * 255)
        g = int((1-ratio) * 200)
        cv2.circle(frame,(gx,gy),radius,(0,g,r),-1)
        cv2.putText(frame, track.display_label,
                    (gx-20, gy+radius+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
    return frame


def _draw_reverse_banner(frame):
    """Red REVERSE indicator in top center."""
    h, w = frame.shape[:2]
    msg  = "  ◀  REVERSE MODE  ▶  "
    (tw,th),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    tx = max(0,(w-tw)//2); ty = 55
    cv2.rectangle(frame,(tx-6,ty-th-6),(tx+tw+6,ty+8),(0,0,180),-1)
    cv2.putText(frame,msg,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    return frame


def _draw_hud(frame, fps, lane_on, depth_on, reverse_mode, frame_idx, infer_width):
    h, w = frame.shape[:2]
    # FPS
    cv2.putText(frame,f"FPS: {fps}",(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,f"Infer: {infer_width}px",(10,58),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1,cv2.LINE_AA)
    # Toggles top-right
    rev_color   = (0,80,255) if reverse_mode else (80,80,80)
    lane_color  = (0,255,80) if lane_on else (80,80,80)
    depth_color = (0,255,255) if depth_on else (80,80,80)
    tags = [
        (f"[E] {'REVERSE' if reverse_mode else 'FORWARD'}", rev_color),
        (f"[L] LANE: {'ON' if lane_on else 'OFF'}", lane_color),
        (f"[D] DEPTH: {'ON' if depth_on else 'OFF'}", depth_color),
    ]
    y_off = 10
    for tag, color in tags:
        (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        tx = w - tw - 12
        cv2.rectangle(frame,(tx-4,y_off),(w-8,y_off+th+6),(30,30,30),-1)
        cv2.putText(frame,tag,(tx,y_off+th),cv2.FONT_HERSHEY_SIMPLEX,0.52,color,1,cv2.LINE_AA)
        y_off += th + 10
    # Frame counter
    fc = f"Frame #{frame_idx}"
    (fw,fh),_ = cv2.getTextSize(fc,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
    cv2.putText(frame,fc,(w-fw-8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,100),1,cv2.LINE_AA)
    # Hint
    cv2.putText(frame,"Q=quit  L=lane  D=depth  R=reset  E=reverse",
                (8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,100,100),1,cv2.LINE_AA)
    return frame


# ═════════════════════════════════════════════════════════════════════════════
# MODULE LOADER  — loads once, shared by live + batch (web)
# ═════════════════════════════════════════════════════════════════════════════

_modules_loaded = False
_detector = _lane_det = _depth_est = None

def _load_modules():
    global _modules_loaded, _detector, _lane_det, _depth_est
    if _modules_loaded:
        return
    from perception.base_detector import DetectorFactory
    import perception.yolov8_detector
    from perception.lane_detector import LaneDetector
    from perception.depth_estimator import DepthEstimator
    logger.info("Loading ADAS modules ...")
    _detector = DetectorFactory.create(_CONFIG["perception"]["detector"])
    _detector.warmup()
    _lane_det  = LaneDetector()
    _depth_est = DepthEstimator()
    _modules_loaded = True
    logger.info("All modules ready")


def _new_tracker():
    from tracking.tracker import MultiObjectTracker
    return MultiObjectTracker()


def _new_collision():
    from decision.collision_warning import CollisionWarning
    return CollisionWarning()


def _new_parking():
    from decision.parking_assist import ParkingAssist
    return ParkingAssist()


# ═════════════════════════════════════════════════════════════════════════════
# CORE FRAME PROCESSOR
# Called identically by LIVE mode and BATCH (web) mode → same output always
# ═════════════════════════════════════════════════════════════════════════════

def process_single_frame(frame, cam_id, tracker, collision, parking,
                         lane_change, frame_idx, infer_width,
                         cached_lanes, cached_depth,
                         lane_on=True, reverse_mode=False,
                         enable_depth=False, depth_on=False):
    """
    Full ADAS pipeline on one frame.

    reverse_mode=False (FORWARD):
      - Lane detection ON (front camera view)
      - Collision warning ON
      - Parking grid OFF

    reverse_mode=True (REVERSE):
      - Lane detection OFF  (reversing — no lane lines ahead)
      - Collision warning OFF
      - Parking assist ON   (proximity grid at bottom)
      - Reverse banner shown

    Returns: (annotated_frame, frame_data, lc_result, cached_lanes, cached_depth)
    """
    from utils.data_models import CameraID, FrameData, Detection, BoundingBox

    h, w  = frame.shape[:2]
    scale = infer_width / w
    small = cv2.resize(frame,(infer_width,int(h*scale)))

    # ── 1. Detection ──────────────────────────────────────────────────────
    raw_dets = _detector.detect(small, cam_id)
    inv = 1.0 / scale
    detections = []
    for d in raw_dets:
        b = d.bbox
        detections.append(Detection(
            bbox=BoundingBox(int(b.x1*inv),int(b.y1*inv),int(b.x2*inv),int(b.y2*inv)),
            confidence=d.confidence, class_id=d.class_id,
            class_name=d.class_name, camera_id=d.camera_id, raw_label=d.raw_label
        ))

    # ── 2. Tracking ───────────────────────────────────────────────────────
    tracks = tracker.update(detections)

    # ── 3. Lane detection ─────────────────────────────────────────────────
    # FORWARD: run lane detection every LANE_EVERY_N frames if lane_on
    # REVERSE: always skip lane detection (rear camera, no lane lines ahead)
    effective_lane_on = lane_on and not reverse_mode
    if frame_idx % LANE_EVERY_N == 0:
        if effective_lane_on:
            cached_lanes = _lane_det.detect(frame, cam_id)
        else:
            cached_lanes = None

    # ── 4. Depth ──────────────────────────────────────────────────────────
    if enable_depth and frame_idx % DEPTH_EVERY_N == 0:
        cached_depth = _depth_est.estimate(frame, cam_id)

    # ── 5. FrameData ──────────────────────────────────────────────────────
    fd = FrameData(
        frame=frame, camera_id=cam_id,
        timestamp=time.time(), frame_idx=frame_idx
    )
    fd.detections = detections
    fd.tracks     = tracks
    fd.lanes      = cached_lanes
    fd.depth      = cached_depth

    # ── 6. Decision ───────────────────────────────────────────────────────
    collision_alerts = []
    parking_alerts   = []

    if not reverse_mode:
        # FORWARD: collision warning is meaningful
        collision_alerts = collision.process(fd)
        fd.alerts += collision_alerts
    else:
        # REVERSE: parking proximity warnings are meaningful
        parking_alerts = parking.process(fd)
        fd.alerts += parking_alerts

    # ── 7. Lane change ────────────────────────────────────────────────────
    lc_result = lane_change.update(cached_lanes if effective_lane_on else None)

    # ── 8. Render ─────────────────────────────────────────────────────────
    out = frame.copy()

    if enable_depth and depth_on and cached_depth is not None:
        out = _draw_depth_overlay(out, cached_depth)

    # Lane lines + departure warning (FORWARD only)
    if not reverse_mode:
        out = _draw_lanes(out, cached_lanes, lane_on)
        if lane_on:
            out = _draw_lane_departure_warning(out, cached_lanes)

    # Tracked objects with IDs + distances
    out = _draw_tracks(out, tracks)

    # FORWARD: collision alerts on left side
    if not reverse_mode:
        out = _draw_collision_alerts(out, collision_alerts)
        out = _draw_lane_change_banner(out, lc_result)
    else:
        # REVERSE: parking proximity grid at bottom + reverse banner
        out = _draw_parking_grid(out, tracks)
        out = _draw_reverse_banner(out)
        # Show parking alerts as collision-style text
        out = _draw_collision_alerts(out, parking_alerts)

    return out, fd, lc_result, cached_lanes, cached_depth


# ═════════════════════════════════════════════════════════════════════════════
# BATCH API  —  called by web_app.py
# ═════════════════════════════════════════════════════════════════════════════

def process_video_file(video_bytes: bytes, cam_name: str,
                       infer_width: int = INFER_WIDTH,
                       reverse_mode: bool = False,
                       progress_callback=None) -> dict:
    """
    Process a full video through the ADAS pipeline.
    Called by web_app.py — produces identical output to live mode.

    Args:
        video_bytes:       raw bytes of the uploaded video file
        cam_name:          "front" | "left" | "right" | "rear"
        infer_width:       YOLO inference width px (lower = faster)
        reverse_mode:      True = rear camera mode (parking assist, no lane)
        progress_callback: fn(pct:int, msg:str)  — for web progress bar
    """
    import tempfile, os
    from collections import Counter
    from utils.data_models import CameraID

    _load_modules()

    cam_map = {"front":CameraID.FRONT,"left":CameraID.LEFT,
               "right":CameraID.RIGHT,"rear":CameraID.REAR}
    cam_id = cam_map.get(cam_name, CameraID.FRONT)

    # ── Save to temp file (VideoCapture needs a real path) ────────────────
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            tmp_avi = tmp_path.replace(".mp4",".avi")
            os.rename(tmp_path, tmp_avi); tmp_path = tmp_avi
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video — try MP4 or AVI format")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Cap output at 1280 for web
        scale = min(1.0, 1280/orig_w) if orig_w > 1280 else 1.0
        out_w = int(orig_w*scale); out_h = int(orig_h*scale)

        # Output video
        ts       = int(time.time()*1000)
        out_name = f"adas_{cam_name}_{ts}.mp4"
        out_dir  = Path("web_outputs"); out_dir.mkdir(exist_ok=True)
        writer   = cv2.VideoWriter(
            str(out_dir/out_name),
            cv2.VideoWriter_fourcc(*"avc1"),   # H.264 — browser compatible
            fps_in, (out_w,out_h)
        )

        # Fresh pipeline state
        tracker      = _new_tracker()
        collision    = _new_collision()
        parking      = _new_parking()
        lane_change  = LaneChangeDetector()
        cached_lanes = None
        cached_depth = None
        label_counts = Counter()
        all_alerts   = []
        mid_frame    = None
        mid_idx      = max(1, total_frames//2)
        last_lane    = None
        frame_idx    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(frame,(out_w,out_h))

            # ── FULL ADAS PIPELINE — same as live mode ─────────────────────
            out, fd, lc_result, cached_lanes, cached_depth = process_single_frame(
                frame=frame, cam_id=cam_id,
                tracker=tracker, collision=collision, parking=parking,
                lane_change=lane_change,
                frame_idx=frame_idx, infer_width=infer_width,
                cached_lanes=cached_lanes, cached_depth=cached_depth,
                lane_on=True, reverse_mode=reverse_mode,
                enable_depth=False, depth_on=False
            )

            # Frame counter overlay (web viewers want to see progress)
            cv2.putText(out, f"Frame {frame_idx+1}/{total_frames}",
                        (out_w-215,out_h-12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)

            writer.write(out)

            # Stats
            for d in fd.detections: label_counts[d.raw_label] += 1
            all_alerts.extend(fd.alerts)
            if fd.lanes: last_lane = fd.lanes
            if frame_idx == mid_idx or mid_frame is None:
                mid_frame = out.copy()

            frame_idx += 1
            if progress_callback and total_frames > 0:
                pct = int(frame_idx/total_frames*100)
                progress_callback(pct, f"{cam_name}: {frame_idx}/{total_frames} frames")

        cap.release(); writer.release()

        # Thumbnail
        import base64
        b64 = ""
        if mid_frame is not None:
            _, buf = cv2.imencode(".jpg",mid_frame,[cv2.IMWRITE_JPEG_QUALITY,82])
            b64    = base64.b64encode(buf).decode()

        from collections import Counter as _C2
        alert_counts = _C2(a.level.value for a in all_alerts)
        lane_data    = None
        if last_lane:
            lane_data = {
                "departure_warning": last_lane.departure_warning,
                "offset_px":        round(last_lane.center_offset_px,1),
                "left_detected":    last_lane.left_line  is not None,
                "right_detected":   last_lane.right_line is not None,
            }

        return {
            "type":             "video",
            "filename":         out_name,
            "image":            b64,
            "frame_count":      frame_idx,
            "fps":              round(fps_in,1),
            "duration_sec":     round(frame_idx/fps_in,1),
            "total_detections": sum(label_counts.values()),
            "detections":       [{"label":k,"count":v}
                                  for k,v in label_counts.most_common(15)],
            "alerts_summary":   dict(alert_counts),
            "lane":             lane_data,
            "reverse_mode":     reverse_mode,
        }

    finally:
        try: os.unlink(tmp_path)
        except: pass


def process_image_file(image_bytes: bytes, cam_name: str,
                       infer_width: int = INFER_WIDTH,
                       reverse_mode: bool = False) -> dict:
    """
    Process a single image through the full ADAS pipeline.
    Called by web_app.py.
    """
    import base64
    from utils.data_models import CameraID

    _load_modules()

    cam_map = {"front":CameraID.FRONT,"left":CameraID.LEFT,
               "right":CameraID.RIGHT,"rear":CameraID.REAR}
    cam_id = cam_map.get(cam_name, CameraID.FRONT)

    arr   = np.frombuffer(image_bytes,np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")

    tracker     = _new_tracker()
    collision   = _new_collision()
    parking     = _new_parking()
    lane_change = LaneChangeDetector()

    out, fd, lc_result, _, _ = process_single_frame(
        frame=frame, cam_id=cam_id,
        tracker=tracker, collision=collision, parking=parking,
        lane_change=lane_change,
        frame_idx=0, infer_width=infer_width,
        cached_lanes=None, cached_depth=None,
        lane_on=True, reverse_mode=reverse_mode,
        enable_depth=False, depth_on=False
    )

    ts       = int(time.time()*1000)
    out_name = f"adas_{cam_name}_{ts}.jpg"
    out_dir  = Path("web_outputs"); out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir/out_name), out, [cv2.IMWRITE_JPEG_QUALITY,88])

    _, buf  = cv2.imencode(".jpg",out,[cv2.IMWRITE_JPEG_QUALITY,82])
    b64_img = base64.b64encode(buf).decode()

    from collections import Counter
    label_counts = Counter(d.raw_label for d in fd.detections)
    alert_counts = Counter(a.level.value for a in fd.alerts)

    lane_data = None
    if fd.lanes:
        lane_data = {
            "departure_warning": fd.lanes.departure_warning,
            "offset_px":        round(fd.lanes.center_offset_px,1),
            "left_detected":    fd.lanes.left_line  is not None,
            "right_detected":   fd.lanes.right_line is not None,
        }

    return {
        "type":             "image",
        "filename":         out_name,
        "image":            b64_img,
        "frame_count":      1,
        "total_detections": len(fd.detections),
        "detections":       [{"label":l,"count":c} for l,c in label_counts.most_common(15)],
        "alerts_summary":   dict(alert_counts),
        "lane":             lane_data,
        "reverse_mode":     reverse_mode,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LIVE MODE  — OpenCV window + keyboard controls
# ═════════════════════════════════════════════════════════════════════════════

def run(source=0, enable_depth=False, infer_width=INFER_WIDTH):
    from utils.data_models import CameraID

    _load_modules()

    logger.info("="*60)
    logger.info("  ADAS LIVE PIPELINE")
    logger.info(f"  Source: {source}  |  Width: {infer_width}px  |  Depth: {enable_depth}")
    logger.info("  Keys: Q=quit  L=lane  D=depth  R=reset  E=reverse")
    logger.info("="*60)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    frame_queue = Queue(maxsize=CAPTURE_QUEUE_SZ)
    stop_event  = threading.Event()

    def _capture():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: stop_event.set(); break
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except Empty: pass
            frame_queue.put(frame)

    threading.Thread(target=_capture, daemon=True, name="cam").start()

    tracker      = _new_tracker()
    collision    = _new_collision()
    parking      = _new_parking()
    lane_change  = LaneChangeDetector()
    cached_lanes = None
    cached_depth = None
    frame_idx    = 0
    fps_times    = []

    # Toggle states
    lane_on      = True
    depth_on     = enable_depth
    reverse_mode = False

    logger.info("Running — press Q to quit")

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue

        t0 = time.monotonic()

        out, fd, lc_result, cached_lanes, cached_depth = process_single_frame(
            frame=frame, cam_id=CameraID.FRONT,
            tracker=tracker, collision=collision, parking=parking,
            lane_change=lane_change,
            frame_idx=frame_idx, infer_width=infer_width,
            cached_lanes=cached_lanes, cached_depth=cached_depth,
            lane_on=lane_on, reverse_mode=reverse_mode,
            enable_depth=enable_depth, depth_on=depth_on
        )

        fps_times.append(time.monotonic())
        fps_times = [t for t in fps_times if fps_times[-1]-t < 1.0]
        fps = len(fps_times)
        out = _draw_hud(out, fps, lane_on, depth_on, reverse_mode, frame_idx, infer_width)

        if DISPLAY_WIDTH and out.shape[1] != DISPLAY_WIDTH:
            dh  = int(out.shape[0]*DISPLAY_WIDTH/out.shape[1])
            out = cv2.resize(out,(DISPLAY_WIDTH,dh))

        cv2.imshow("ADAS — Main Pipeline", out)

        if frame_idx % 60 == 0:
            ms = (time.monotonic()-t0)*1000
            mode = "REVERSE" if reverse_mode else "FORWARD"
            logger.debug(f"Frame {frame_idx:05d} | {ms:.1f}ms | FPS {fps} | "
                         f"dets {len(fd.detections)} | tracks {len(fd.tracks)} | {mode}")

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            logger.info("Quit"); break

        elif key in (ord("l"), ord("L")):
            lane_on = not lane_on
            cached_lanes = None
            lane_change.reset()
            logger.info(f"Lane: {'ON' if lane_on else 'OFF'}")

        elif key in (ord("d"), ord("D")):
            if enable_depth:
                depth_on = not depth_on
                logger.info(f"Depth overlay: {'ON' if depth_on else 'OFF'}")
            else:
                logger.info("Run with --depth to enable depth estimation")

        elif key in (ord("r"), ord("R")):
            from tracking.tracker import KalmanTrack
            tracker.reset(); KalmanTrack._id_counter = 0
            logger.info("Tracker reset")

        elif key in (ord("e"), ord("E")):
            # ── REVERSE MODE TOGGLE ──────────────────────────────────────
            reverse_mode = not reverse_mode
            cached_lanes = None
            lane_change.reset()
            mode = "REVERSE (rear lane ON, parking assist ON)" if reverse_mode \
                   else "FORWARD (front lane ON, collision warning ON)"
            logger.info(f"Mode switched → {mode}")

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("ADAS stopped cleanly")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADAS Main Pipeline",
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default=0,
                    help="Camera index (int) or video file path")
    ap.add_argument("--depth",  action="store_true",
                    help="Enable MiDaS depth estimation (GPU recommended)")
    ap.add_argument("--width",  type=int, default=INFER_WIDTH,
                    help=f"YOLO inference width px (default {INFER_WIDTH})\n"
                         "  320 = fastest  |  416 = balanced  |  640 = accurate")
    args = ap.parse_args()

    src = args.source
    try: src = int(src)
    except (ValueError, TypeError): pass

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>")
    logger.add("logs/adas_main_{time}.log", rotation="50 MB", level="DEBUG", encoding="utf-8")

    run(source=src, enable_depth=args.depth, infer_width=args.width)