"""
main.py  —  ADAS Full Pipeline  (Single Camera + All Features)
═══════════════════════════════════════════════════════════════

Features:
  ✓ Object detection  (YOLOv8 — all 80 COCO classes, correct labels)
  ✓ Lane detection    (Hough lines with filled polygon overlay)
  ✓ Lane CHANGE detection  (watches center offset shift over time)
  ✓ Lane departure warning (when car drifts out of lane)
  ✓ Collision warning (Time-To-Collision + distance for every tracked object)
  ✓ Object tracking   (Kalman + Hungarian — stable IDs across frames)
  ✓ Depth estimation  (MiDaS — optional, off by default on CPU)
  ✓ Lane ON/OFF toggle (press L key to toggle lane overlay)
  ✓ Threaded camera capture (no blocking, always live frame)
  ✓ FPS counter on screen

Keyboard controls:
  Q  — Quit
  L  — Toggle lane detection overlay ON / OFF
  D  — Toggle depth map overlay ON / OFF  (only works if --depth enabled)
  R  — Reset tracker (clears all track IDs)

Run:
  python main.py                          # webcam 0, no depth
  python main.py --source 1               # second webcam
  python main.py --source video.mp4       # video file
  python main.py --depth                  # enable MiDaS depth (GPU recommended)
  python main.py --width 320              # faster inference
  python main.py --width 640              # more accurate
"""

import argparse
import sys
import time
import threading
import collections
from queue import Queue, Empty

import cv2
import numpy as np
from loguru import logger

# ── Inline config — no yaml file needed to run main.py directly ──────────────
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
            "backend": "yolov8",
            "model_path": "yolov8n.pt",
            "confidence": 0.40,
            "iou_threshold": 0.45,
            "classes": None,
            "device": "auto"
        },
        "lane_detector": {"backend": "heuristic", "roi_top_ratio": 0.55},
        "depth_estimator": {
            "backend": "midas",
            "model_type": "MiDaS_small",
            "enabled": False
        }
    },
    "tracking": {
        "backend": "deepsort",
        "max_age": 30,
        "min_hits": 2,
        "iou_threshold": 0.3
    },
    "decision": {
        "collision":  {
            "ttc_warning_sec":  3.0,
            "ttc_critical_sec": 1.5,
            "min_object_area":  2000
        },
        "blind_spot": {"zone_x_ratio": 0.25, "min_velocity_kmh": 10.0},
        "parking":    {
            "critical_distance_cm": 40,
            "warning_distance_cm":  100,
            "grid_cells": [10, 6]
        }
    },
    "output": {
        "display": {"font_scale": 0.55, "overlay_alpha": 0.4},
        "audio":   {"enabled": False, "beep_cooldown_sec": 2.0},
        "logger":  {
            "enabled": True,
            "log_dir": "logs/",
            "record_video": False,
            "video_dir": "logs/video/"
        }
    }
}

# ── Performance knobs ─────────────────────────────────────────────────────────
INFER_WIDTH      = 416    # Width for YOLO inference. 320=fast, 416=balanced, 640=accurate
DEPTH_EVERY_N    = 10     # Run MiDaS every N frames (heavy on CPU)
LANE_EVERY_N     = 3      # Run lane detection every N frames
DISPLAY_WIDTH    = 1024   # Final display window width (0 = no resize)
CAPTURE_QUEUE_SZ = 2      # Camera buffer (keep small = low latency)

# ── Lane change detection params ──────────────────────────────────────────────
LANE_CHANGE_HISTORY  = 15   # frames to keep offset history
LANE_CHANGE_THRESH   = 40   # px shift in offset = lane change detected
LANE_CHANGE_COOLDOWN = 60   # frames between consecutive lane change alerts

# ── Color palette (BGR) ───────────────────────────────────────────────────────
_C = {
    "car":          (0,   200, 255),  "truck":     (0,   140, 255),
    "bus":          (0,   100, 200),  "motorcycle":(255, 150,   0),
    "bicycle":      (255, 200,   0),  "airplane":  (180, 180, 255),
    "train":        (120, 100, 200),  "boat":      (200, 220, 255),
    "person":       (100, 255, 100),
    "dog":          (255,  80, 200),  "cat":       (255, 100, 220),
    "horse":        (210,  80, 180),  "bird":      (255, 160, 200),
    "sheep":        (220, 130, 200),  "cow":       (200,  80, 160),
    "elephant":     (180,  60, 140),  "bear":      (160,  40, 120),
    "zebra":        (240, 100, 200),  "giraffe":   (230, 120, 190),
    "traffic light":(0,   255, 255),  "stop sign": (0,    50, 255),
    "fire hydrant": (0,   100, 255),
    "backpack":     (200, 200,  80),  "umbrella":  (180, 200, 100),
    "suitcase":     (140, 160,  60),  "laptop":    (200, 160, 100),
    "cell phone":   (180, 140,  80),
    "unknown":      (200, 200, 200),  "default":   (200, 200, 200),
}

def _color(label: str):
    return _C.get(label.lower(), _C["default"])


# ═════════════════════════════════════════════════════════════════════════════
# LANE CHANGE DETECTOR
# Tracks the center_offset over time and detects a sudden directional shift.
# ═════════════════════════════════════════════════════════════════════════════

class LaneChangeDetector:
    """
    Watches the lane center_offset across frames.
    When offset shifts significantly in one direction → lane change detected.

    center_offset > 0  means car is RIGHT of lane center
    center_offset < 0  means car is LEFT  of lane center

    A sudden shift from positive → negative (or vice versa) over a short
    window = crossing the lane marking.
    """

    def __init__(self):
        self._history      = collections.deque(maxlen=LANE_CHANGE_HISTORY)
        self._cooldown     = 0          # frames remaining before next alert allowed
        self._last_dir     = None       # "left" or "right"
        self.active        = False      # is a lane change currently happening?
        self.direction     = ""         # "left" or "right"
        self.active_frames = 0          # how many frames the change has been active

    def update(self, lanes) -> dict:
        """
        Feed current LaneInfo. Returns dict:
          {
            "changing": bool,
            "direction": "left" | "right" | "",
            "alert_text": str,   # human readable, empty if no change
          }
        """
        result = {"changing": False, "direction": "", "alert_text": ""}

        if lanes is None or (lanes.left_line is None and lanes.right_line is None):
            self._history.clear()
            self.active = False
            self._cooldown = max(0, self._cooldown - 1)
            return result

        offset = lanes.center_offset_px
        self._history.append(offset)

        if self._cooldown > 0:
            self._cooldown -= 1

        if len(self._history) < LANE_CHANGE_HISTORY // 2:
            return result   # not enough data yet

        # Compare first half vs second half of history window
        half       = len(self._history) // 2
        old_vals   = list(self._history)[:half]
        new_vals   = list(self._history)[half:]
        old_avg    = sum(old_vals) / len(old_vals)
        new_avg    = sum(new_vals) / len(new_vals)
        shift      = new_avg - old_avg   # positive = shifted right, negative = shifted left

        changing = abs(shift) > LANE_CHANGE_THRESH

        if changing and self._cooldown == 0:
            direction = "right" if shift > 0 else "left"
            self.active        = True
            self.direction     = direction
            self.active_frames = LANE_CHANGE_COOLDOWN // 2
            self._cooldown     = LANE_CHANGE_COOLDOWN
            self._last_dir     = direction

            result["changing"]   = True
            result["direction"]  = direction
            result["alert_text"] = f"LANE CHANGE  {direction.upper()}"
        else:
            # Decay active state
            if self.active_frames > 0:
                self.active_frames -= 1
                result["changing"]   = True
                result["direction"]  = self.direction
                result["alert_text"] = f"LANE CHANGE  {self.direction.upper()}"
            else:
                self.active = False
                self.direction = ""

        return result


# ═════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS  — all drawing is here, never scattered in the loop
# ═════════════════════════════════════════════════════════════════════════════

def _draw_tracks(frame, tracks, font_scale=0.55):
    for track in tracks:
        b     = track.bbox
        label = track.display_label
        color = _color(label)
        thick = 2 if track.confidence > 0.6 else 1

        # Bounding box
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, thick)

        # Label tag: "#ID label CONF%"
        conf_pct = int(track.confidence * 100)
        tag      = f"#{track.track_id} {label} {conf_pct}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tag_y = max(b.y1, th + 8)
        cv2.rectangle(frame, (b.x1, tag_y - th - 6), (b.x1 + tw + 6, tag_y), color, -1)
        cv2.putText(frame, tag, (b.x1 + 3, tag_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Distance below box
        if track.estimated_distance_m is not None:
            dist_str = f"{track.estimated_distance_m:.1f}m"
            cx = (b.x1 + b.x2) // 2
            cv2.putText(frame, dist_str, (cx - 20, b.y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame


def _draw_lanes(frame, lanes, lane_on: bool):
    """
    Draw lane lines + filled polygon between them.
    If lane_on is False — draws nothing at all (toggle OFF).
    """
    if not lane_on or lanes is None:
        return frame

    h, w   = frame.shape[:2]
    overlay = frame.copy()
    has_left  = lanes.left_line  is not None
    has_right = lanes.right_line is not None

    # Draw filled polygon between the two lane lines (green tint)
    if has_left and has_right:
        lx1, ly1, lx2, ly2 = lanes.left_line
        rx1, ry1, rx2, ry2 = lanes.right_line
        pts = np.array([[lx1, ly1], [lx2, ly2],
                         [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 180, 0))

    # Draw individual lane lines
    if has_left:
        lx1, ly1, lx2, ly2 = lanes.left_line
        cv2.line(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 0), 4)
    if has_right:
        rx1, ry1, rx2, ry2 = lanes.right_line
        cv2.line(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 0), 4)

    # Red border flash on departure
    if lanes.departure_warning:
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)


def _draw_collision_alerts(frame, alerts, font_scale=0.7):
    """
    Draw collision + distance alerts on left side of frame.
    CRITICAL = red,  WARNING = orange,  INFO = yellow
    """
    alert_colors = {
        "critical": (0,   0,   255),
        "warning":  (0,   165, 255),
        "info":     (0,   220, 255),
    }
    y = 90
    for alert in alerts:
        level = alert.level.value.lower()
        color = alert_colors.get(level, (255, 255, 255))
        msg   = alert.message

        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        # Dark semi-transparent background
        cv2.rectangle(frame, (6, y - th - 5), (10 + tw + 6, y + 6), (20, 20, 20), -1)
        cv2.putText(frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        y += th + 16
    return frame


def _draw_lane_change_banner(frame, lc_result: dict):
    """
    Big centered banner for lane change.
    Shows direction arrow + text.
    Fades in/out via alpha blend.
    """
    if not lc_result.get("changing"):
        return frame

    h, w      = frame.shape[:2]
    direction = lc_result.get("direction", "")
    arrow     = "<<  " if direction == "left" else "  >>"
    text      = f"{arrow}  LANE CHANGE {direction.upper()}  {arrow}"
    overlay   = frame.copy()

    # Full-width amber banner
    banner_h = 64
    banner_y = h // 2 - banner_h // 2
    cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 140, 255), -1)

    # Centered text
    fs = 1.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    tx = max(0, (w - tw) // 2)
    ty = banner_y + (banner_h + th) // 2
    cv2.putText(overlay, text, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)


def _draw_lane_departure_warning(frame, lanes):
    """
    Separate departure warning text (different from lane change).
    Shows when car has already drifted out of lane center.
    """
    if lanes is None or not lanes.departure_warning:
        return frame
    h, w = frame.shape[:2]
    msg  = "  LANE DEPARTURE WARNING  "
    fs   = 0.85
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = max(0, (w - tw) // 2)
    ty = 55
    cv2.rectangle(frame, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 8), (0, 0, 200), -1)
    cv2.putText(frame, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _draw_hud(frame, fps: int, lane_on: bool, depth_on: bool,
              frame_idx: int, infer_width: int):
    """
    Top-right HUD: FPS, mode indicators, keyboard hints.
    """
    h, w = frame.shape[:2]

    # ── FPS (top left, green) ──────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Infer size ─────────────────────────────────────────────────────────
    cv2.putText(frame, f"Infer: {infer_width}px", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Mode toggles (top right) ───────────────────────────────────────────
    lane_color  = (0, 255,  80) if lane_on  else (80, 80, 80)
    depth_color = (0, 255, 255) if depth_on else (80, 80, 80)

    lane_text  = "[L] LANE: ON " if lane_on  else "[L] LANE: OFF"
    depth_text = "[D] DEPTH: ON" if depth_on else "[D] DEPTH: OFF"

    (lw, lh), _ = cv2.getTextSize(lane_text,  cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    (dw, dh), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    lx = w - lw - 10
    dx = w - dw - 10

    cv2.rectangle(frame, (lx - 4, 8),       (w - 6, 8 + lh + 6),        (30, 30, 30), -1)
    cv2.rectangle(frame, (dx - 4, 8+lh+10), (w - 6, 8+lh+10+dh+6),     (30, 30, 30), -1)

    cv2.putText(frame, lane_text,  (lx, 8 + lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, lane_color, 1, cv2.LINE_AA)
    cv2.putText(frame, depth_text, (dx, 8 + lh + 10 + dh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, depth_color, 1, cv2.LINE_AA)

    # ── Frame counter (bottom right) ───────────────────────────────────────
    fc_text = f"Frame #{frame_idx}"
    (fw, fh), _ = cv2.getTextSize(fc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, fc_text, (w - fw - 8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Keyboard hint (bottom left) ────────────────────────────────────────
    hint = "Q=quit  L=lane  D=depth  R=reset"
    cv2.putText(frame, hint, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


def _draw_depth_overlay(frame, depth_map):
    """
    Overlay colorized depth map as semi-transparent layer.
    Only shown when depth toggle is ON and depth is available.
    """
    if depth_map is None or depth_map.map is None:
        return frame
    d = depth_map.map
    d_norm = (d * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)
    d_resized = cv2.resize(d_color, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.65, d_resized, 0.35, 0)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run(source=0, enable_depth=False, infer_width=INFER_WIDTH):

    # ── Inject config before any module imports ────────────────────────────
    import utils.config_loader as cfg_mod
    _CONFIG["cameras"]["front"]["source"] = source
    _CONFIG["perception"]["depth_estimator"]["enabled"] = enable_depth
    cfg_mod._config = _CONFIG

    # ── Imports (after config is injected) ────────────────────────────────
    from perception.base_detector import DetectorFactory
    import perception.yolov8_detector          # registers "yolov8" backend
    from perception.lane_detector import LaneDetector
    from perception.depth_estimator import DepthEstimator
    from tracking.tracker import MultiObjectTracker
    from decision.collision_warning import CollisionWarning
    from utils.data_models import (
        CameraID, FrameData, Detection, BoundingBox
    )

    logger.info("=" * 55)
    logger.info("  ADAS MAIN PIPELINE")
    logger.info(f"  Source      : {source}")
    logger.info(f"  Infer width : {infer_width}px")
    logger.info(f"  Depth       : {'ON' if enable_depth else 'OFF'}")
    logger.info(f"  Keys        : Q=quit  L=lane  D=depth  R=reset")
    logger.info("=" * 55)

    # ── Module initialisation ─────────────────────────────────────────────
    detector      = DetectorFactory.create(_CONFIG["perception"]["detector"])
    lane_det      = LaneDetector()
    depth_est     = DepthEstimator()
    tracker       = MultiObjectTracker()
    collision     = CollisionWarning()
    lane_change   = LaneChangeDetector()

    logger.info("Warming up YOLO (first inference is slow) ...")
    detector.warmup()
    logger.info("Warmup done — starting camera")

    # ── Camera capture thread ─────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    frame_queue = Queue(maxsize=CAPTURE_QUEUE_SZ)
    stop_event  = threading.Event()

    def _capture_loop():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera read failed — stopping")
                stop_event.set()
                break
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put(frame)

    threading.Thread(target=_capture_loop, daemon=True, name="camera").start()

    # ── State ─────────────────────────────────────────────────────────────
    frame_idx    = 0
    fps_times    = []
    cached_lanes = None
    cached_depth = None

    # Toggle states
    lane_on  = True          # L key
    depth_on = enable_depth  # D key (only useful if --depth was passed)

    logger.info("Running — press Q in the window to quit")

    # ── Main loop ─────────────────────────────────────────────────────────
    while not stop_event.is_set():

        # ── Get frame ─────────────────────────────────────────────────────
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue

        t0   = time.monotonic()
        h, w = frame.shape[:2]

        # ── 1. Resize frame for faster YOLO inference ─────────────────────
        scale = infer_width / w
        small = cv2.resize(frame, (infer_width, int(h * scale)))

        # ── 2. Object detection ───────────────────────────────────────────
        raw_dets = detector.detect(small, CameraID.FRONT)

        # Scale bounding boxes back to original resolution
        inv        = 1.0 / scale
        detections = []
        for d in raw_dets:
            b = d.bbox
            detections.append(Detection(
                bbox=BoundingBox(
                    int(b.x1 * inv), int(b.y1 * inv),
                    int(b.x2 * inv), int(b.y2 * inv)
                ),
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                camera_id=d.camera_id,
                raw_label=d.raw_label          # always carry through
            ))

        # ── 3. Object tracking ────────────────────────────────────────────
        tracks = tracker.update(detections)

        # ── 4. Lane detection (every LANE_EVERY_N frames) ─────────────────
        if frame_idx % LANE_EVERY_N == 0:
            if lane_on:
                cached_lanes = lane_det.detect(frame, CameraID.FRONT)
            else:
                cached_lanes = None   # don't waste CPU if toggled off

        # ── 5. Depth estimation (optional, every DEPTH_EVERY_N frames) ────
        if enable_depth and frame_idx % DEPTH_EVERY_N == 0:
            cached_depth = depth_est.estimate(frame, CameraID.FRONT)

        # ── 6. Build FrameData ────────────────────────────────────────────
        fd = FrameData(
            frame=frame,
            camera_id=CameraID.FRONT,
            timestamp=time.time(),
            frame_idx=frame_idx
        )
        fd.detections = detections
        fd.tracks     = tracks
        fd.lanes      = cached_lanes
        fd.depth      = cached_depth

        # ── 7. Collision warnings ─────────────────────────────────────────
        collision_alerts = collision.process(fd)
        fd.alerts       += collision_alerts

        # ── 8. Lane change detection ──────────────────────────────────────
        lc_result = lane_change.update(cached_lanes if lane_on else None)

        # ── 9. RENDER ─────────────────────────────────────────────────────
        out = frame.copy()

        # 9a. Depth overlay (behind everything else)
        if enable_depth and depth_on and cached_depth is not None:
            out = _draw_depth_overlay(out, cached_depth)

        # 9b. Lane polygon + lines
        out = _draw_lanes(out, cached_lanes, lane_on)

        # 9c. Lane departure warning (bottom-of-lane area text)
        if lane_on:
            out = _draw_lane_departure_warning(out, cached_lanes)

        # 9d. Tracked object bounding boxes
        out = _draw_tracks(out, tracks)

        # 9e. Collision alert text (left side)
        out = _draw_collision_alerts(out, collision_alerts)

        # 9f. Lane CHANGE banner (big centered, only when changing)
        out = _draw_lane_change_banner(out, lc_result)

        # 9g. HUD (FPS, toggles, hints)
        fps_times.append(time.monotonic())
        fps_times = [t for t in fps_times if fps_times[-1] - t < 1.0]
        fps = len(fps_times)

        out = _draw_hud(out, fps, lane_on, depth_on, frame_idx, infer_width)

        # ── 10. Resize display window if needed ───────────────────────────
        if DISPLAY_WIDTH and out.shape[1] != DISPLAY_WIDTH:
            dh  = int(out.shape[0] * DISPLAY_WIDTH / out.shape[1])
            out = cv2.resize(out, (DISPLAY_WIDTH, dh))

        cv2.imshow("ADAS — Main Pipeline", out)

        # ── 11. Debug log every 60 frames ─────────────────────────────────
        if frame_idx % 60 == 0:
            ms = (time.monotonic() - t0) * 1000
            logger.debug(
                f"Frame {frame_idx:05d} | {ms:5.1f}ms | "
                f"FPS {fps:2d} | dets {len(detections):2d} | "
                f"tracks {len(tracks):2d} | alerts {len(fd.alerts):2d}"
            )

        frame_idx += 1

        # ── 12. Keyboard input ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:       # Q or ESC = quit
            logger.info("Quit key pressed")
            break

        elif key == ord("l") or key == ord("L"):   # L = toggle lane
            lane_on = not lane_on
            cached_lanes = None                    # clear stale lanes immediately
            lane_change  = LaneChangeDetector()    # reset lane change state
            status = "ON" if lane_on else "OFF"
            logger.info(f"Lane detection toggled: {status}")

        elif key == ord("d") or key == ord("D"):   # D = toggle depth overlay
            if enable_depth:
                depth_on = not depth_on
                status = "ON" if depth_on else "OFF"
                logger.info(f"Depth overlay toggled: {status}")
            else:
                logger.info("Depth not enabled — run with --depth flag to use")

        elif key == ord("r") or key == ord("R"):   # R = reset tracker
            from tracking.tracker import KalmanTrack
            tracker.reset()
            KalmanTrack._id_counter = 0
            logger.info("Tracker reset — all track IDs cleared")

    # ── Cleanup ───────────────────────────────────────────────────────────
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("ADAS pipeline stopped cleanly")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ADAS Main Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--source", default=0,
        help="Camera index (0,1,2...) or path to video file\n"
             "Example: --source 0  or  --source road.mp4"
    )
    ap.add_argument(
        "--depth", action="store_true",
        help="Enable MiDaS depth estimation\n"
             "Recommended only on GPU — very slow on CPU"
    )
    ap.add_argument(
        "--width", type=int, default=INFER_WIDTH,
        help=f"YOLO inference width in pixels (default: {INFER_WIDTH})\n"
             f"  320 = fastest (~25 FPS CPU)\n"
             f"  416 = balanced (~15 FPS CPU)  ← default\n"
             f"  640 = most accurate (~8 FPS CPU)"
    )
    args = ap.parse_args()

    # Convert source to int if it's a digit string
    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass   # keep as string (video file path)

    # Set up logger
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(
        "logs/adas_main_{time}.log",
        rotation="50 MB",
        level="DEBUG",
        encoding="utf-8"
    )

    run(source=src, enable_depth=args.depth, infer_width=args.width)