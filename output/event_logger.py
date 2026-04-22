"""
output/event_logger.py
───────────────────────
Logs all ADAS events (alerts, detections, track stats) to JSON files.
Optionally records annotated video to disk.

Log format (one JSON object per line — easy to parse, grep, stream):
  {"ts": 1714000000.0, "camera": "front", "event": "collision_warning",
   "level": "critical", "message": "...", "track_id": 5, "ttc": 1.2}
"""

import json
import time
from pathlib import Path
from typing import Optional
import cv2
from loguru import logger

from utils.data_models import Alert, CameraID, FrameData
from utils.config_loader import get_config


class EventLogger:

    def __init__(self):
        cfg = get_config()["output"]["logger"]
        self.enabled       = cfg.get("enabled", True)
        self.log_dir       = Path(cfg.get("log_dir", "logs/"))
        self.record_video  = cfg.get("record_video", False)
        self.video_dir     = Path(cfg.get("video_dir", "logs/video/"))

        self._log_file     = None
        self._video_writers: dict = {}

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = self.log_dir / f"adas_{ts}.jsonl"
            self._log_file = open(log_path, "w")
            logger.info(f"Event log: {log_path}")

        if self.record_video:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def log(self, frame_data: FrameData, annotated_frame=None):
        if not self.enabled:
            return

        for alert in frame_data.alerts:
            record = {
                "ts":       frame_data.timestamp,
                "frame":    frame_data.frame_idx,
                "camera":   frame_data.camera_id.value,
                "event":    alert.source,
                "level":    alert.level.value,
                "message":  alert.message,
                "track_id": alert.track_id,
                "ttc":      alert.ttc,
            }
            self._log_file.write(json.dumps(record) + "\n")

        if self.record_video and annotated_frame is not None:
            self._write_video(frame_data.camera_id, annotated_frame)

    def _write_video(self, camera_id: CameraID, frame):
        if camera_id not in self._video_writers:
            h, w = frame.shape[:2]
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = str(self.video_dir / f"{camera_id.value}_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writers[camera_id] = cv2.VideoWriter(
                path, fourcc, 30, (w, h))
            logger.info(f"Video recording: {path}")

        self._video_writers[camera_id].write(frame)

    def flush(self):
        if self._log_file:
            self._log_file.flush()

    def close(self):
        if self._log_file:
            self._log_file.close()
        for writer in self._video_writers.values():
            writer.release()
