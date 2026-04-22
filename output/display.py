"""
output/display.py — Fixed version
Fixed: uses track.display_label (raw YOLO name) instead of track.class_name.value
       Added colors for all animal/object classes
       Added confidence-based box thickness
"""

import time
from typing import Dict, List, Optional
import cv2
import numpy as np

from utils.data_models import (
    Alert, AlertLevel, CameraID, FrameData, LaneInfo, Track
)
from utils.config_loader import get_config


# ── Color palette (BGR) — covers all 80 COCO classes by category ──────────
COLORS = {
    # Vehicles
    "car":          (0,   200, 255),
    "truck":        (0,   140, 255),
    "bus":          (0,   100, 200),
    "motorcycle":   (255, 150,   0),
    "bicycle":      (255, 200,   0),
    "airplane":     (180, 180, 255),
    "train":        (120, 100, 200),
    "boat":         (200, 220, 255),

    # People
    "person":       (100, 255, 100),

    # Animals — all shown in pink/magenta so you instantly see them
    "dog":          (255,  80, 200),
    "cat":          (255, 100, 220),
    "horse":        (210,  80, 180),
    "bird":         (255, 160, 200),
    "sheep":        (220, 130, 200),
    "cow":          (200,  80, 160),
    "elephant":     (180,  60, 140),
    "bear":         (160,  40, 120),
    "zebra":        (240, 100, 200),
    "giraffe":      (230, 120, 190),

    # Traffic signs / infrastructure
    "traffic light":  (0,   255, 255),
    "stop sign":      (0,    50, 255),
    "fire hydrant":   (0,   100, 255),
    "parking meter":  (60,  180, 255),
    "bench":          (160, 220, 180),

    # Common objects
    "backpack":     (200, 200,  80),
    "umbrella":     (180, 200, 100),
    "handbag":      (160, 180,  80),
    "suitcase":     (140, 160,  60),
    "sports ball":  (80,  200, 200),
    "bottle":       (100, 200, 180),
    "chair":        (180, 180, 140),
    "couch":        (160, 160, 120),
    "bed":          (140, 140, 100),
    "laptop":       (200, 160, 100),
    "cell phone":   (180, 140,  80),
    "tv":           (160, 120,  60),

    # Fallback
    "unknown":      (200, 200, 200),
    "default":      (200, 200, 200),

    # Lane / alert colors
    "lane_left":    (0,   255,   0),
    "lane_right":   (0,   255,   0),
    "warning":      (0,   200, 255),
    "critical":     (0,     0, 255),
    "info":         (255, 200,   0),
}

ALERT_COLORS = {
    AlertLevel.CRITICAL: (0,   0,   255),
    AlertLevel.WARNING:  (0,   165, 255),
    AlertLevel.INFO:     (255, 200,   0),
    AlertLevel.NONE:     (255, 255, 255),
}


class DisplayManager:

    def __init__(self):
        cfg = get_config()
        self.show   = cfg["system"].get("show_display", True)
        self._scale = cfg["output"]["display"].get("font_scale", 0.55)
        self._fps_tracker: Dict[CameraID, list] = {}

    def render(self, frame_data: FrameData) -> np.ndarray:
        frame = frame_data.frame.copy()
        frame = self._draw_tracks(frame, frame_data.tracks)
        frame = self._draw_lanes(frame, frame_data.lanes)
        frame = self._draw_alerts(frame, frame_data.alerts)
        return frame

    def show_frame(self, camera_id: CameraID, frame: np.ndarray):
        if not self.show:
            return
        cv2.imshow(f"ADAS — {camera_id.value}", frame)

    def check_quit(self) -> bool:
        return cv2.waitKey(1) & 0xFF == ord("q")

    def cleanup(self):
        cv2.destroyAllWindows()

    def _get_color(self, label: str):
        """Look up color by label string — case insensitive, falls back to default."""
        return COLORS.get(label.lower(), COLORS["default"])

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        for track in tracks:
            b = track.bbox

            # ── CRITICAL FIX: use display_label not class_name.value ───────
            label_text = track.display_label   # "dog", "cat", "person" etc — ALWAYS correct
            color = self._get_color(label_text)

            # Box thickness scales with confidence
            thickness = 2 if track.confidence > 0.6 else 1

            # Bounding box
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, thickness)

            # Label: "#ID  label  conf%"
            conf_pct = int(track.confidence * 100)
            display  = f"#{track.track_id} {label_text} {conf_pct}%"

            (lw, lh), baseline = cv2.getTextSize(
                display, cv2.FONT_HERSHEY_SIMPLEX, self._scale, 1)

            # Label background — don't draw outside frame top
            label_y = max(b.y1, lh + 6)
            cv2.rectangle(frame,
                          (b.x1, label_y - lh - 6),
                          (b.x1 + lw + 6, label_y),
                          color, -1)

            # Dark text on colored background
            text_color = (0, 0, 0)
            cv2.putText(frame, display,
                        (b.x1 + 3, label_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, self._scale, text_color, 1, cv2.LINE_AA)

            # Distance label below box if available
            if track.estimated_distance_m is not None:
                dist_str = f"{track.estimated_distance_m:.1f}m"
                cv2.putText(frame, dist_str,
                            (b.bottom_center[0] - 20, b.bottom_center[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, self._scale, color, 2, cv2.LINE_AA)

        return frame

    def _draw_lanes(self, frame: np.ndarray,
                    lanes: Optional[LaneInfo]) -> np.ndarray:
        if lanes is None:
            return frame
        overlay = frame.copy()
        if lanes.left_line is not None:
            x1, y1, x2, y2 = lanes.left_line
            cv2.line(overlay, (x1, y1), (x2, y2), COLORS["lane_left"], 4)
        if lanes.right_line is not None:
            x1, y1, x2, y2 = lanes.right_line
            cv2.line(overlay, (x1, y1), (x2, y2), COLORS["lane_right"], 4)
        if lanes.departure_warning:
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(overlay, "LANE DEPARTURE", (w // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    def _draw_alerts(self, frame: np.ndarray,
                     alerts: List[Alert]) -> np.ndarray:
        y_offset = 80
        for alert in sorted(alerts, key=lambda a: a.level.value, reverse=True):
            color = ALERT_COLORS.get(alert.level, (255, 255, 255))
            (tw, th), _ = cv2.getTextSize(
                alert.message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame,
                          (8, y_offset - th - 4),
                          (8 + tw + 8, y_offset + 4),
                          (30, 30, 30), -1)
            cv2.putText(frame, alert.message,
                        (12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y_offset += th + 14
        return frame