"""
cameras/camera_manager.py
──────────────────────────
Manages all camera streams.
Each camera runs in its own thread → frames arrive without blocking the pipeline.

Design: Producer-Consumer pattern.
  Thread per camera → puts frames in a queue → pipeline consumes from queue.
"""

import threading
import time
from collections import defaultdict
from queue import Queue, Empty
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from utils.config_loader import get_config
from utils.data_models import CameraID, FrameData


class CameraStream:
    """
    A single camera stream running in its own thread.
    Reads frames from OpenCV VideoCapture and enqueues them.
    """

    def __init__(self, camera_id: CameraID, source, resolution: Tuple[int, int], fps: int):
        self.camera_id  = camera_id
        self.source     = source
        self.resolution = resolution
        self.target_fps = fps
        self.frame_idx  = 0

        self._queue: Queue = Queue(maxsize=3)   # small buffer = low latency
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"[{self.camera_id}] Cannot open source: {self.source}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._running = True
        self._thread = threading.Thread(target=self._read_loop,
                                        name=f"cam-{self.camera_id}",
                                        daemon=True)
        self._thread.start()
        logger.info(f"[{self.camera_id}] Stream started (source={self.source})")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info(f"[{self.camera_id}] Stream stopped")

    def _read_loop(self):
        frame_duration = 1.0 / self.target_fps
        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"[{self.camera_id}] Frame read failed — retrying")
                time.sleep(0.1)
                continue

            frame_data = FrameData(
                frame=frame,
                camera_id=self.camera_id,
                timestamp=time.time(),
                frame_idx=self.frame_idx
            )
            self.frame_idx += 1

            # Drop oldest frame if queue full (keep latency low)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame_data)

            elapsed = time.monotonic() - t0
            sleep_t = max(0, frame_duration - elapsed)
            time.sleep(sleep_t)

    def get_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running


class CameraManager:
    """
    Manages all active camera streams.
    Provides unified interface: manager.get_frames() → dict of latest frames.

    Adding a 5th camera: add entry to config/settings.yaml → zero code changes.
    """

    def __init__(self):
        self._streams: Dict[CameraID, CameraStream] = {}
        cfg = get_config()["cameras"]

        for cam_name, cam_cfg in cfg.items():
            if not cam_cfg.get("enabled", True):
                continue
            cam_id = CameraID(cam_name)
            stream = CameraStream(
                camera_id=cam_id,
                source=cam_cfg["source"],
                resolution=tuple(cam_cfg["resolution"]),
                fps=cam_cfg.get("fps", 30)
            )
            self._streams[cam_id] = stream

    def start_all(self):
        started = []
        for cam_id, stream in self._streams.items():
            if stream.start():
                started.append(cam_id)
        logger.info(f"Camera Manager: {len(started)} streams active: {started}")

    def stop_all(self):
        for stream in self._streams.values():
            stream.stop()

    def get_frames(self) -> Dict[CameraID, Optional[FrameData]]:
        """Returns latest frame from each camera. None if not available."""
        return {cam_id: stream.get_frame()
                for cam_id, stream in self._streams.items()}

    def get_frame(self, camera_id: CameraID) -> Optional[FrameData]:
        stream = self._streams.get(camera_id)
        return stream.get_frame() if stream else None

    @property
    def active_cameras(self):
        return list(self._streams.keys())
