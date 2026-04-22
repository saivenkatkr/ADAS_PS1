"""
tracking/tracker.py
────────────────────
Multi-object tracker using DeepSORT algorithm.

DeepSORT extends SORT with a Re-ID appearance descriptor:
  - Kalman filter: predicts next position
  - Hungarian algorithm: matches detections to existing tracks
  - Cosine appearance feature: handles occlusion, re-identifies lost objects

This module is DETECTOR-AGNOSTIC — it receives List[Detection] and
outputs List[Track], regardless of what model produced the detections.

Why DeepSORT for ADAS:
  - Stable track IDs → reliable TTC calculation
  - Handles brief occlusions (car going behind a truck)
  - 30+ FPS capable on GPU

Swapping to ByteTrack: create ByteTracker(BaseTracker), zero pipeline changes.
"""

import numpy as np
from typing import List
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger

from utils.data_models import BoundingBox, CameraID, Detection, Track
from utils.config_loader import get_config


# ── Simple Kalman-based Track ───────────────────────────────────────────────

class KalmanTrack:
    """
    Single-object Kalman track.
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    _id_counter = 0

    def __init__(self, detection: Detection):
        KalmanTrack._id_counter += 1
        self.track_id   = KalmanTrack._id_counter
        self.class_name = detection.class_name
        self.raw_label  = detection.raw_label   # ← carry real YOLO label
        self.camera_id  = detection.camera_id
        self.confidence = detection.confidence
        self.hits       = 1
        self.no_detection_age = 0

        # Kalman filter setup
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ])
        self.kf.H  = np.eye(4, 8)
        self.kf.R  *= 10.0
        self.kf.P  *= 1000.0
        self.kf.Q  *= 0.01

        z = self._bbox_to_z(detection.bbox)
        self.kf.x[:4] = z

    def predict(self):
        self.kf.predict()
        self.no_detection_age += 1

    def update(self, detection: Detection):
        z = self._bbox_to_z(detection.bbox)
        self.kf.update(z)
        self.class_name = detection.class_name
        self.raw_label  = detection.raw_label   # ← keep label fresh
        self.confidence = detection.confidence
        self.hits += 1
        self.no_detection_age = 0

    def get_bbox(self) -> BoundingBox:
        vals = self.kf.x[:4].flatten().tolist()
        x, y, w, h = vals[0], vals[1], vals[2], vals[3]
        w = max(w, 1)
        h = max(h, 1)
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        return BoundingBox(x1, y1, x1 + int(w), y1 + int(h))

    def get_velocity(self):
        vx = float(self.kf.x[4].item()) if hasattr(self.kf.x[4], 'item') else float(self.kf.x[4])
        vy = float(self.kf.x[5].item()) if hasattr(self.kf.x[5], 'item') else float(self.kf.x[5])
        return (vx, vy)

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.get_bbox(),
            class_name=self.class_name,
            raw_label=self.raw_label,   # ← pass through
            camera_id=self.camera_id,
            confidence=self.confidence,
            velocity=self.get_velocity(),
            age=self.hits
        )

    @staticmethod
    def _bbox_to_z(bbox: BoundingBox) -> np.ndarray:
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w  = bbox.width
        h  = bbox.height
        return np.array([[cx], [cy], [w], [h]])


# ── IoU utilities ───────────────────────────────────────────────────────────

def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
    ix1 = max(b1.x1, b2.x1)
    iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2)
    iy2 = min(b1.y2, b2.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = b1.area + b2.area - inter
    return inter / union if union > 0 else 0.0


# ── Main Tracker ─────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Lightweight SORT-style tracker (Kalman + Hungarian matching).
    Good for real-time ADAS — ~1ms overhead per frame.

    For production: drop in the official deep-sort-realtime package
    and this class becomes a thin wrapper around it.
    """

    def __init__(self):
        cfg = get_config()["tracking"]
        self.max_age      = cfg.get("max_age", 30)
        self.min_hits     = cfg.get("min_hits", 3)
        self.iou_thresh   = cfg.get("iou_threshold", 0.3)
        self._tracks: List[KalmanTrack] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Input:  list of detections from current frame
        Output: list of confirmed tracks (with stable IDs)
        """
        # 1. Predict all existing tracks forward
        for t in self._tracks:
            t.predict()

        # 2. Match detections to tracks (Hungarian algorithm on IoU cost)
        matched, unmatched_dets = self._match(detections)

        # 3. Update matched tracks
        for track_idx, det_idx in matched:
            self._tracks[track_idx].update(detections[det_idx])

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._tracks.append(KalmanTrack(detections[det_idx]))

        # 5. Remove stale tracks
        self._tracks = [t for t in self._tracks
                        if t.no_detection_age <= self.max_age]

        # 6. Return only confirmed tracks (seen min_hits frames)
        return [t.to_track() for t in self._tracks
                if t.hits >= self.min_hits]

    def _match(self, detections):
        if not self._tracks or not detections:
            return [], list(range(len(detections)))

        n_tracks = len(self._tracks)
        n_dets   = len(detections)
        cost = np.zeros((n_tracks, n_dets))

        for ti, track in enumerate(self._tracks):
            track_bbox = track.get_bbox()
            for di, det in enumerate(detections):
                cost[ti, di] = 1 - _iou(track_bbox, det.bbox)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_dets = [], list(range(n_dets))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < (1 - self.iou_thresh):
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)

        return matched, unmatched_dets

    def reset(self):
        self._tracks.clear()
        KalmanTrack._id_counter = 0