"""
decision/collision_warning.py
──────────────────────────────
Fixed: ObjectClass.ANIMAL removed — replaced with all individual COCO animal classes.
"""

from typing import List, Dict, Optional
from loguru import logger

from utils.data_models import (
    Alert, AlertLevel, CameraID, FrameData, ObjectClass, Track
)
from utils.config_loader import get_config

# Real-world heights (meters) — used for focal-length distance estimation
_OBJECT_HEIGHTS_M: Dict[ObjectClass, float] = {
    # Vehicles
    ObjectClass.CAR:        1.5,
    ObjectClass.TRUCK:      3.5,
    ObjectClass.BUS:        3.2,
    ObjectClass.MOTORCYCLE: 1.2,
    ObjectClass.BICYCLE:    1.1,
    ObjectClass.TRAIN:      4.0,
    ObjectClass.BOAT:       2.0,
    ObjectClass.AIRPLANE:   5.0,
    # People
    ObjectClass.PERSON:     1.7,
    # Animals — every COCO animal is a real collision risk on roads
    ObjectClass.DOG:        0.6,
    ObjectClass.CAT:        0.3,
    ObjectClass.HORSE:      1.6,
    ObjectClass.COW:        1.4,
    ObjectClass.SHEEP:      0.9,
    ObjectClass.ELEPHANT:   3.0,
    ObjectClass.BEAR:       1.2,
    ObjectClass.ZEBRA:      1.5,
    ObjectClass.GIRAFFE:    4.5,
    ObjectClass.BIRD:       0.3,
    # Fallback for anything not listed
    ObjectClass.UNKNOWN:    1.5,
}

_FOCAL_LENGTH_PX = 700.0   # calibrate for your actual camera


class CollisionWarning:
    """
    Issues collision warnings for objects in the FRONT camera.
    Uses Time-To-Collision (TTC) estimated from bbox size + track velocity.
    """

    def __init__(self):
        cfg = get_config()["decision"]["collision"]
        self.ttc_warning  = cfg.get("ttc_warning_sec",  3.0)
        self.ttc_critical = cfg.get("ttc_critical_sec", 1.5)
        self.min_obj_area = cfg.get("min_object_area",  2000)

    def process(self, frame_data: FrameData) -> List[Alert]:
        if frame_data.camera_id != CameraID.FRONT:
            return []

        alerts = []
        frame_h = frame_data.frame.shape[0]

        for track in frame_data.tracks:
            if track.bbox.area < self.min_obj_area:
                continue

            dist_m = self._estimate_distance(track, frame_h)
            ttc    = self._compute_ttc(track, dist_m)
            track.estimated_distance_m = dist_m

            alert = self._evaluate(track, dist_m, ttc)
            if alert:
                alerts.append(alert)

        return alerts

    def _estimate_distance(self, track: Track, frame_h: int) -> float:
        real_h = _OBJECT_HEIGHTS_M.get(track.class_name, 1.5)
        bbox_h = max(track.bbox.height, 1)
        return round((real_h * _FOCAL_LENGTH_PX) / bbox_h, 2)

    def _compute_ttc(self, track: Track, dist_m: float) -> float:
        _, vy = track.velocity
        if vy >= 0:
            return float("inf")
        closing_speed_ms = abs(vy) * 0.05
        if closing_speed_ms < 0.01:
            return float("inf")
        return dist_m / closing_speed_ms

    def _evaluate(self, track: Track, dist_m: float, ttc: float) -> Optional[Alert]:
        cam   = track.camera_id
        label = track.display_label   # use display_label — always correct

        if ttc < self.ttc_critical:
            return Alert(
                level=AlertLevel.CRITICAL,
                source="collision_warning",
                message=f"BRAKE! {label} {dist_m:.1f}m (TTC {ttc:.1f}s)",
                camera_id=cam,
                track_id=track.track_id,
                ttc=ttc
            )
        elif ttc < self.ttc_warning:
            return Alert(
                level=AlertLevel.WARNING,
                source="collision_warning",
                message=f"Warning: {label} {dist_m:.1f}m ahead (TTC {ttc:.1f}s)",
                camera_id=cam,
                track_id=track.track_id,
                ttc=ttc
            )
        elif dist_m < 5.0:
            return Alert(
                level=AlertLevel.INFO,
                source="collision_warning",
                message=f"Close: {label} at {dist_m:.1f}m",
                camera_id=cam,
                track_id=track.track_id
            )
        return None