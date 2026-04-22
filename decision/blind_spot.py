"""
decision/blind_spot.py
───────────────────────
Blind spot detection using LEFT and RIGHT side cameras.

Defines a blind spot zone in each side camera frame.
If a vehicle/motorcycle is detected within the zone → alert driver.

Zone definition:
  The "dangerous" zone is when an object is:
  - Large enough (close proximity) — bbox width > threshold
  - Moving roughly parallel (vx small, vy near 0)
  - In the near-side portion of the frame

Used for:
  - Lane change safety decision
  - Warning before merge
"""

from typing import List

from utils.data_models import (
    Alert, AlertLevel, CameraID, FrameData, ObjectClass
)
from utils.config_loader import get_config


_VEHICLE_CLASSES = {
    ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS,
    ObjectClass.MOTORCYCLE, ObjectClass.BICYCLE
}


class BlindSpotDetector:
    """
    Processes LEFT and RIGHT camera frames.
    Returns alerts when vehicles detected in blind zone.
    """

    def __init__(self):
        cfg = get_config()["decision"]["blind_spot"]
        self.zone_x_ratio      = cfg.get("zone_x_ratio", 0.25)
        self.min_velocity_kmh  = cfg.get("min_velocity_kmh", 10.0)

    def process(self, frame_data: FrameData) -> List[Alert]:
        if frame_data.camera_id not in (CameraID.LEFT, CameraID.RIGHT):
            return []

        frame_w = frame_data.frame.shape[1]
        alerts  = []
        side    = frame_data.camera_id

        for track in frame_data.tracks:
            if track.class_name not in _VEHICLE_CLASSES:
                continue

            in_zone = self._in_blind_zone(track, frame_w, side)
            if in_zone:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    source="blind_spot",
                    message=f"Vehicle in {side.value} blind spot!",
                    camera_id=side,
                    track_id=track.track_id
                ))

        return alerts

    def _in_blind_zone(self, track, frame_w: int, side: CameraID) -> bool:
        """
        Blind zone = near side of the frame (car-side edge).
        LEFT camera:  zone is RIGHT portion of frame (x > frame_w * (1 - zone_ratio))
        RIGHT camera: zone is LEFT portion of frame  (x < frame_w * zone_ratio)
        """
        cx = track.bbox.center[0]
        if side == CameraID.LEFT:
            in_zone = cx > frame_w * (1 - self.zone_x_ratio)
        else:  # RIGHT
            in_zone = cx < frame_w * self.zone_x_ratio

        # Object must be a minimum size (not a tiny distant vehicle)
        large_enough = track.bbox.width > frame_w * 0.08
        return in_zone and large_enough
