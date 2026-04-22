"""
tests/test_core.py
───────────────────
Unit tests for core ADAS modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from utils.data_models import (
    Alert, AlertLevel, BoundingBox, CameraID, Detection,
    FrameData, LaneInfo, ObjectClass, Track
)


# ── BoundingBox tests ────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_dimensions(self):
        bb = BoundingBox(10, 20, 110, 70)
        assert bb.width  == 100
        assert bb.height == 50
        assert bb.area   == 5000

    def test_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.center == (50, 50)

    def test_bottom_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.bottom_center == (50, 100)


# ── ObjectClass mapping ───────────────────────────────────────────────────────

class TestObjectClass:
    def test_yolo_id_mapping(self):
        assert ObjectClass.from_yolo_id(0)  == ObjectClass.PERSON
        assert ObjectClass.from_yolo_id(2)  == ObjectClass.CAR
        assert ObjectClass.from_yolo_id(7)  == ObjectClass.TRUCK
        assert ObjectClass.from_yolo_id(99) == ObjectClass.UNKNOWN


# ── FrameData pipeline contract ───────────────────────────────────────────────

class TestFrameData:
    def _make_frame(self, cam=CameraID.FRONT) -> FrameData:
        return FrameData(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            camera_id=cam,
            timestamp=1000.0,
            frame_idx=0
        )

    def test_default_lists_empty(self):
        fd = self._make_frame()
        assert fd.detections == []
        assert fd.tracks == []
        assert fd.alerts == []
        assert fd.lanes is None
        assert fd.depth is None

    def test_alerts_append(self):
        fd = self._make_frame()
        fd.alerts.append(Alert(
            level=AlertLevel.WARNING,
            source="test",
            message="test alert",
            camera_id=CameraID.FRONT
        ))
        assert len(fd.alerts) == 1
        assert fd.alerts[0].level == AlertLevel.WARNING


# ── CollisionWarning tests ────────────────────────────────────────────────────

class TestCollisionWarning:
    def _make_track(self, y_velocity: float, bbox_height: int,
                    cls=ObjectClass.CAR) -> Track:
        return Track(
            track_id=1,
            bbox=BoundingBox(300, 400, 500, 400 + bbox_height),
            class_name=cls,
            camera_id=CameraID.FRONT,
            confidence=0.9,
            velocity=(0.0, y_velocity)
        )

    def _make_frame_data(self, tracks) -> FrameData:
        fd = FrameData(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            camera_id=CameraID.FRONT,
            timestamp=1000.0,
            frame_idx=0
        )
        fd.tracks = tracks
        return fd

    @patch("decision.collision_warning.get_config")
    def test_no_alert_for_far_object(self, mock_cfg):
        mock_cfg.return_value = {
            "decision": {"collision": {
                "ttc_warning_sec": 3.0,
                "ttc_critical_sec": 1.5,
                "min_object_area": 100
            }}
        }
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # Small bbox = far object, not approaching (vy=0)
        track = self._make_track(y_velocity=0.0, bbox_height=40)
        fd = self._make_frame_data([track])
        alerts = cw.process(fd)
        assert all(a.level != AlertLevel.CRITICAL for a in alerts)

    @patch("decision.collision_warning.get_config")
    def test_critical_alert_for_close_fast_object(self, mock_cfg):
        mock_cfg.return_value = {
            "decision": {"collision": {
                "ttc_warning_sec": 3.0,
                "ttc_critical_sec": 1.5,
                "min_object_area": 100
            }}
        }
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # Large bbox = close object, high negative vy = fast approach
        track = self._make_track(y_velocity=-50.0, bbox_height=300)
        fd = self._make_frame_data([track])
        alerts = cw.process(fd)
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) >= 1


# ── LaneDetector tests ────────────────────────────────────────────────────────

class TestLaneDetector:
    @patch("perception.lane_detector.get_config")
    def test_detect_returns_lane_info(self, mock_cfg):
        mock_cfg.return_value = {
            "perception": {"lane_detector": {
                "backend": "heuristic",
                "roi_top_ratio": 0.55
            }}
        }
        from perception.lane_detector import LaneDetector
        from utils.data_models import LaneInfo
        det = LaneDetector()

        # Plain gray frame — no lines detectable
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame, CameraID.FRONT)
        assert isinstance(result, LaneInfo)
        assert result.camera_id == CameraID.FRONT

    @patch("perception.lane_detector.get_config")
    def test_white_lines_detected(self, mock_cfg):
        mock_cfg.return_value = {
            "perception": {"lane_detector": {
                "backend": "heuristic",
                "roi_top_ratio": 0.55
            }}
        }
        from perception.lane_detector import LaneDetector
        det = LaneDetector()

        # Draw white diagonal lines on black frame (simulate lane markings)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.line(frame, (100, 480), (280, 260), (255, 255, 255), 5)  # left lane
        cv2.line(frame, (540, 480), (360, 260), (255, 255, 255), 5)  # right lane

        result = det.detect(frame, CameraID.FRONT)
        # At least one line should be detected
        assert result.left_line is not None or result.right_line is not None


# ── BlindSpotDetector tests ───────────────────────────────────────────────────

class TestBlindSpotDetector:
    @patch("decision.blind_spot.get_config")
    def test_vehicle_in_left_blind_zone(self, mock_cfg):
        mock_cfg.return_value = {
            "decision": {"blind_spot": {
                "zone_x_ratio": 0.25,
                "min_velocity_kmh": 10.0
            }}
        }
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        # Vehicle on far-right of left camera = in blind zone
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track = Track(
            track_id=1,
            bbox=BoundingBox(1100, 300, 1250, 500),  # right side of frame
            class_name=ObjectClass.CAR,
            camera_id=CameraID.LEFT,
            confidence=0.9
        )
        fd = FrameData(frame=frame, camera_id=CameraID.LEFT,
                       timestamp=0.0, frame_idx=0)
        fd.tracks = [track]
        alerts = bsd.process(fd)
        assert len(alerts) == 1
        assert alerts[0].source == "blind_spot"

    @patch("decision.blind_spot.get_config")
    def test_no_alert_for_front_camera(self, mock_cfg):
        mock_cfg.return_value = {
            "decision": {"blind_spot": {
                "zone_x_ratio": 0.25,
                "min_velocity_kmh": 10.0
            }}
        }
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        fd = FrameData(frame=frame, camera_id=CameraID.FRONT,
                       timestamp=0.0, frame_idx=0)
        fd.tracks = [Track(
            track_id=1,
            bbox=BoundingBox(100, 100, 300, 300),
            class_name=ObjectClass.CAR,
            camera_id=CameraID.FRONT,
            confidence=0.9
        )]
        alerts = bsd.process(fd)
        assert len(alerts) == 0   # front camera → blind spot doesn't apply


# ── Tracker tests ─────────────────────────────────────────────────────────────

class TestTracker:
    def _make_detection(self, x1, y1, x2, y2,
                        cls=ObjectClass.CAR) -> Detection:
        return Detection(
            bbox=BoundingBox(x1, y1, x2, y2),
            confidence=0.9,
            class_id=2,
            class_name=cls,
            camera_id=CameraID.FRONT
        )

    def test_track_ids_stable(self):
        from tracking.tracker import MultiObjectTracker, KalmanTrack
        KalmanTrack._id_counter = 0

        with patch("tracking.tracker.get_config") as mock_cfg:
            mock_cfg.return_value = {
                "tracking": {
                    "max_age": 30,
                    "min_hits": 1,
                    "iou_threshold": 0.3
                }
            }
            tracker = MultiObjectTracker()
            det = self._make_detection(100, 100, 200, 200)

            tracks1 = tracker.update([det])
            tracks2 = tracker.update([det])

            assert len(tracks1) == 1
            assert len(tracks2) == 1
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_empty_detections(self):
        from tracking.tracker import MultiObjectTracker
        with patch("tracking.tracker.get_config") as mock_cfg:
            mock_cfg.return_value = {
                "tracking": {
                    "max_age": 30, "min_hits": 3, "iou_threshold": 0.3
                }
            }
            tracker = MultiObjectTracker()
            result = tracker.update([])
            assert result == []
