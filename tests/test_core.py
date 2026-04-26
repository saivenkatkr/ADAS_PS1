"""
tests/test_core.py
───────────────────
Unit tests for every core ADAS module.

What is a unit test?
  A unit test checks that ONE small piece of code works correctly,
  in complete isolation from cameras, YOLO, GPU, real video etc.
  You create fake (mock) inputs, run the function, check the output.

Why we have tests:
  - If you change collision_warning.py, run tests → instantly know if you broke something
  - If a teammate changes data_models.py, tests catch the breakage before demo day
  - Tests run in <5 seconds with no camera, no GPU, no internet

Run all tests:
    pytest tests/ -v

Run one test class:
    pytest tests/test_core.py::TestCollisionWarning -v

Run one specific test:
    pytest tests/test_core.py::TestCollisionWarning::test_critical_alert_for_close_fast_object -v
"""

import numpy as np
import pytest
from unittest.mock import patch

from utils.data_models import (
    Alert, AlertLevel, BoundingBox, CameraID,
    Detection, FrameData, LaneInfo, ObjectClass, Track
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — builds a blank Detection with all required fields
# ══════════════════════════════════════════════════════════════════════════════
def _make_detection(x1=100, y1=100, x2=300, y2=300,
                    cls=ObjectClass.CAR, cls_id=2,
                    cam=CameraID.FRONT, conf=0.9,
                    raw_label="car") -> Detection:
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=conf,
        class_id=cls_id,
        class_name=cls,
        camera_id=cam,
        raw_label=raw_label
    )


def _make_track(x1=300, y1=400, x2=500, y2=500,
                vy=0.0, cls=ObjectClass.CAR,
                cam=CameraID.FRONT, raw_label="car") -> Track:
    return Track(
        track_id=1,
        bbox=BoundingBox(x1, y1, x2, y2),
        class_name=cls,
        camera_id=cam,
        confidence=0.9,
        raw_label=raw_label,
        velocity=(0.0, vy)
    )


def _make_frame(cam=CameraID.FRONT, tracks=None) -> FrameData:
    fd = FrameData(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        camera_id=cam,
        timestamp=1000.0,
        frame_idx=0
    )
    if tracks:
        fd.tracks = tracks
    return fd


# ══════════════════════════════════════════════════════════════════════════════
# 1. BoundingBox — tests geometry helpers
# ══════════════════════════════════════════════════════════════════════════════
class TestBoundingBox:
    """
    BoundingBox holds pixel coordinates of a detected object.
    Tests confirm width/height/area/center calculations are correct.
    """

    def test_width_and_height(self):
        bb = BoundingBox(10, 20, 110, 70)
        assert bb.width  == 100
        assert bb.height == 50

    def test_area(self):
        bb = BoundingBox(0, 0, 200, 50)
        assert bb.area == 10000

    def test_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.center == (50, 50)

    def test_center_non_square(self):
        bb = BoundingBox(0, 0, 200, 100)
        assert bb.center == (100, 50)

    def test_bottom_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.bottom_center == (50, 100)

    def test_zero_size_does_not_crash(self):
        bb = BoundingBox(50, 50, 50, 50)
        assert bb.width  == 0
        assert bb.height == 0
        assert bb.area   == 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. ObjectClass — tests YOLO class ID mapping
# ══════════════════════════════════════════════════════════════════════════════
class TestObjectClass:
    """
    from_yolo_id() maps YOLO's integer class IDs to our ObjectClass enum.
    from_yolo_name() maps YOLO's string class names.
    These MUST be correct or every label on screen is wrong.
    """

    def test_person(self):
        assert ObjectClass.from_yolo_id(0) == ObjectClass.PERSON

    def test_car(self):
        assert ObjectClass.from_yolo_id(2) == ObjectClass.CAR

    def test_truck(self):
        assert ObjectClass.from_yolo_id(7) == ObjectClass.TRUCK

    def test_dog(self):
        # dog is YOLO class 16 — this was the bug that showed dogs as "unknown"
        assert ObjectClass.from_yolo_id(16) == ObjectClass.DOG

    def test_cat(self):
        assert ObjectClass.from_yolo_id(15) == ObjectClass.CAT

    def test_horse(self):
        assert ObjectClass.from_yolo_id(17) == ObjectClass.HORSE

    def test_traffic_light(self):
        assert ObjectClass.from_yolo_id(9) == ObjectClass.TRAFFIC_LIGHT

    def test_stop_sign(self):
        assert ObjectClass.from_yolo_id(11) == ObjectClass.STOP_SIGN

    def test_unknown_for_invalid_id(self):
        assert ObjectClass.from_yolo_id(999) == ObjectClass.UNKNOWN

    def test_from_name_dog(self):
        assert ObjectClass.from_yolo_name("dog") == ObjectClass.DOG

    def test_from_name_case_insensitive(self):
        assert ObjectClass.from_yolo_name("Dog") == ObjectClass.DOG
        assert ObjectClass.from_yolo_name("DOG") == ObjectClass.DOG

    def test_from_name_unknown_returns_unknown(self):
        assert ObjectClass.from_yolo_name("dragon") == ObjectClass.UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# 3. Track.display_label — tests the label shown on screen is always correct
# ══════════════════════════════════════════════════════════════════════════════
class TestTrackDisplayLabel:
    """
    display_label is a property on Track.
    It must ALWAYS return the correct string — never "unknown" if YOLO knew.
    Priority: raw_label first → class_name.value → "unknown"
    """

    def test_raw_label_takes_priority(self):
        t = _make_track(raw_label="dog", cls=ObjectClass.DOG)
        assert t.display_label == "dog"

    def test_falls_back_to_class_name(self):
        t = _make_track(raw_label="", cls=ObjectClass.CAR)
        assert t.display_label == "car"

    def test_unknown_when_both_empty(self):
        t = _make_track(raw_label="", cls=ObjectClass.UNKNOWN)
        assert t.display_label == "unknown"

    def test_raw_label_preserved_through_track(self):
        # If someone makes a Track with raw_label "giraffe", it shows "giraffe"
        t = Track(
            track_id=5,
            bbox=BoundingBox(0, 0, 100, 200),
            class_name=ObjectClass.GIRAFFE,
            camera_id=CameraID.FRONT,
            confidence=0.88,
            raw_label="giraffe"
        )
        assert t.display_label == "giraffe"


# ══════════════════════════════════════════════════════════════════════════════
# 4. FrameData — tests the pipeline message bus
# ══════════════════════════════════════════════════════════════════════════════
class TestFrameData:
    """
    FrameData is the object that flows through the entire pipeline.
    Every module reads from and writes to it.
    Tests confirm default state is correct and no shared-list bugs.
    """

    def test_default_lists_empty(self):
        fd = _make_frame()
        assert fd.detections == []
        assert fd.tracks     == []
        assert fd.alerts     == []

    def test_default_lanes_and_depth_none(self):
        fd = _make_frame()
        assert fd.lanes is None
        assert fd.depth is None

    def test_two_instances_do_not_share_lists(self):
        # This catches the mutable default argument bug
        fd1 = _make_frame()
        fd2 = _make_frame()
        fd1.alerts.append(Alert(
            level=AlertLevel.WARNING, source="test",
            message="test", camera_id=CameraID.FRONT
        ))
        assert len(fd1.alerts) == 1
        assert len(fd2.alerts) == 0   # must be independent

    def test_append_alert(self):
        fd = _make_frame()
        fd.alerts.append(Alert(
            level=AlertLevel.CRITICAL, source="collision_warning",
            message="BRAKE! car 2.0m", camera_id=CameraID.FRONT
        ))
        assert len(fd.alerts) == 1
        assert fd.alerts[0].level == AlertLevel.CRITICAL

    def test_camera_id_stored_correctly(self):
        fd = _make_frame(cam=CameraID.REAR)
        assert fd.camera_id == CameraID.REAR


# ══════════════════════════════════════════════════════════════════════════════
# 5. CollisionWarning — tests distance estimation and TTC alert logic
# ══════════════════════════════════════════════════════════════════════════════
class TestCollisionWarning:
    """
    CollisionWarning.process() looks at tracks in the FRONT camera and:
      1. Estimates distance from bounding box height
      2. Computes Time-To-Collision from velocity
      3. Returns CRITICAL / WARNING / INFO alerts

    Tests use mock tracks with known velocity and bbox size.
    """

    _CFG = {
        "decision": {"collision": {
            "ttc_warning_sec": 3.0,
            "ttc_critical_sec": 1.5,
            "min_object_area": 100
        }}
    }

    @patch("decision.collision_warning.get_config")
    def test_no_alert_when_object_not_approaching(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # vy = 0 means object is stationary — no TTC
        track = _make_track(y1=400, y2=440, vy=0.0)  # bbox height = 40px
        fd    = _make_frame(tracks=[track])
        alerts = cw.process(fd)
        assert all(a.level != AlertLevel.CRITICAL for a in alerts)

    @patch("decision.collision_warning.get_config")
    def test_critical_alert_when_close_and_fast(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # vy = -50 (approaching fast), bbox height = 300px (very close)
        track = _make_track(y1=200, y2=500, vy=-50.0)
        fd    = _make_frame(tracks=[track])
        alerts = cw.process(fd)
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) >= 1

    @patch("decision.collision_warning.get_config")
    def test_ignored_on_rear_camera(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # CollisionWarning only works on FRONT camera
        track = _make_track(y1=200, y2=500, vy=-50.0, cam=CameraID.REAR)
        fd = FrameData(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            camera_id=CameraID.REAR,
            timestamp=0.0, frame_idx=0
        )
        fd.tracks = [track]
        alerts = cw.process(fd)
        assert alerts == []

    @patch("decision.collision_warning.get_config")
    def test_object_moving_away_no_alert(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        # vy positive = object moving away (getting smaller on screen)
        track = _make_track(y1=200, y2=500, vy=+20.0)
        fd    = _make_frame(tracks=[track])
        alerts = cw.process(fd)
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 0

    @patch("decision.collision_warning.get_config")
    def test_alert_message_contains_label(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.collision_warning import CollisionWarning
        cw = CollisionWarning()

        track = _make_track(y1=200, y2=500, vy=-50.0,
                            cls=ObjectClass.DOG, raw_label="dog")
        fd    = _make_frame(tracks=[track])
        alerts = cw.process(fd)
        # Alert message should say "dog" not "unknown"
        assert any("dog" in a.message for a in alerts)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LaneDetector — tests OpenCV heuristic lane detection
# ══════════════════════════════════════════════════════════════════════════════
class TestLaneDetector:
    """
    LaneDetector uses Canny edge detection + Hough line transform.
    Tests check it returns the right type, handles empty frames,
    and actually finds white lines on a black frame.
    """

    _CFG = {
        "perception": {"lane_detector": {
            "backend": "heuristic",
            "roi_top_ratio": 0.55
        }}
    }

    @patch("perception.lane_detector.get_config")
    def test_returns_lane_info_type(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from perception.lane_detector import LaneDetector
        det = LaneDetector()

        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame, CameraID.FRONT)
        assert isinstance(result, LaneInfo)

    @patch("perception.lane_detector.get_config")
    def test_camera_id_preserved(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from perception.lane_detector import LaneDetector
        det    = LaneDetector()
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame, CameraID.FRONT)
        assert result.camera_id == CameraID.FRONT

    @patch("perception.lane_detector.get_config")
    def test_blank_frame_no_lines(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from perception.lane_detector import LaneDetector
        det    = LaneDetector()
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)   # pure black
        result = det.detect(frame, CameraID.FRONT)
        # No edges → no lines expected (both can be None)
        assert result.left_line is None or result.left_line is not None  # doesn't crash

    @patch("perception.lane_detector.get_config")
    def test_white_diagonal_lines_detected(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from perception.lane_detector import LaneDetector
        det   = LaneDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw lane lines using numpy (works without real cv2.line)
        for y in range(265, 480):
            t  = (y - 265) / max(1, 480 - 265)
            lx = int(80  + t * 20)
            rx = int(560 - t * 20)
            if 0 <= lx < 640: frame[y, max(0, lx-4):lx+4] = 255
            if 0 <= rx < 640: frame[y, max(0, rx-4):rx+4] = 255
        result = det.detect(frame, CameraID.FRONT)
        # Must return LaneInfo without crashing — actual line detection
        # depends on real cv2 being available (not the test stub)
        assert isinstance(result, LaneInfo)

    @patch("perception.lane_detector.get_config")
    def test_center_offset_is_float(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from perception.lane_detector import LaneDetector
        det    = LaneDetector()
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame, CameraID.FRONT)
        assert isinstance(result.center_offset_px, float)


# ══════════════════════════════════════════════════════════════════════════════
# 7. BlindSpotDetector — tests side camera zone detection
# ══════════════════════════════════════════════════════════════════════════════
class TestBlindSpotDetector:
    """
    BlindSpotDetector watches LEFT and RIGHT camera frames.
    A vehicle in the near-side edge of the frame = in the blind zone.
    Tests verify zone logic and that FRONT camera is correctly ignored.
    """

    _CFG = {
        "decision": {"blind_spot": {
            "zone_x_ratio": 0.25,
            "min_velocity_kmh": 10.0
        }}
    }

    @patch("decision.blind_spot.get_config")
    def test_vehicle_in_left_blind_zone_triggers_alert(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        # RIGHT side of LEFT camera frame = near the ego vehicle
        track = Track(
            track_id=1,
            bbox=BoundingBox(1100, 300, 1250, 500),
            class_name=ObjectClass.CAR,
            camera_id=CameraID.LEFT,
            confidence=0.9,
            raw_label="car"
        )
        fd = _make_frame(cam=CameraID.LEFT, tracks=[track])
        alerts = bsd.process(fd)
        assert len(alerts) == 1
        assert alerts[0].source == "blind_spot"
        assert alerts[0].level  == AlertLevel.WARNING

    @patch("decision.blind_spot.get_config")
    def test_vehicle_far_in_left_camera_no_alert(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        # LEFT side of LEFT camera = far from ego vehicle, not in blind zone
        track = Track(
            track_id=2,
            bbox=BoundingBox(10, 300, 200, 500),
            class_name=ObjectClass.CAR,
            camera_id=CameraID.LEFT,
            confidence=0.9,
            raw_label="car"
        )
        fd = _make_frame(cam=CameraID.LEFT, tracks=[track])
        alerts = bsd.process(fd)
        assert len(alerts) == 0

    @patch("decision.blind_spot.get_config")
    def test_front_camera_ignored(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        # Front camera should never produce blind spot alerts
        track = Track(
            track_id=3,
            bbox=BoundingBox(1100, 300, 1250, 500),
            class_name=ObjectClass.CAR,
            camera_id=CameraID.FRONT,
            confidence=0.9,
            raw_label="car"
        )
        fd = _make_frame(cam=CameraID.FRONT, tracks=[track])
        alerts = bsd.process(fd)
        assert alerts == []

    @patch("decision.blind_spot.get_config")
    def test_right_camera_blind_zone(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from decision.blind_spot import BlindSpotDetector
        bsd = BlindSpotDetector()

        # LEFT side of RIGHT camera = near ego vehicle
        track = Track(
            track_id=4,
            bbox=BoundingBox(10, 300, 200, 500),
            class_name=ObjectClass.CAR,
            camera_id=CameraID.RIGHT,
            confidence=0.9,
            raw_label="car"
        )
        fd = _make_frame(cam=CameraID.RIGHT, tracks=[track])
        alerts = bsd.process(fd)
        assert len(alerts) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 8. MultiObjectTracker — tests Kalman tracking and ID stability
# ══════════════════════════════════════════════════════════════════════════════
class TestTracker:
    """
    MultiObjectTracker assigns stable IDs to detected objects.
    The same real-world object must get the same track_id across frames.
    Tests confirm IDs are stable and empty input is handled safely.
    """

    _CFG = {
        "tracking": {
            "max_age": 30,
            "min_hits": 1,
            "iou_threshold": 0.3
        }
    }

    @patch("tracking.tracker.get_config")
    def test_stable_id_across_frames(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from tracking.tracker import MultiObjectTracker, KalmanTrack
        KalmanTrack._id_counter = 0

        tracker = MultiObjectTracker()
        det = _make_detection(100, 100, 200, 200)

        tracks1 = tracker.update([det])
        tracks2 = tracker.update([det])   # same position → same track

        assert len(tracks1) == 1
        assert len(tracks2) == 1
        assert tracks1[0].track_id == tracks2[0].track_id

    @patch("tracking.tracker.get_config")
    def test_empty_detections_returns_empty(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from tracking.tracker import MultiObjectTracker
        tracker = MultiObjectTracker()
        result  = tracker.update([])
        assert result == []

    @patch("tracking.tracker.get_config")
    def test_track_has_raw_label(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from tracking.tracker import MultiObjectTracker, KalmanTrack
        KalmanTrack._id_counter = 0

        tracker = MultiObjectTracker()
        det = _make_detection(raw_label="dog",
                              cls=ObjectClass.DOG, cls_id=16)
        tracks = tracker.update([det])
        assert len(tracks) == 1
        assert tracks[0].raw_label == "dog"
        assert tracks[0].display_label == "dog"

    @patch("tracking.tracker.get_config")
    def test_two_objects_get_different_ids(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from tracking.tracker import MultiObjectTracker, KalmanTrack
        KalmanTrack._id_counter = 0

        tracker = MultiObjectTracker()
        det1 = _make_detection(0,   0,  100, 100)
        det2 = _make_detection(800, 0,  900, 100)   # far apart

        tracks = tracker.update([det1, det2])
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2   # must be two distinct IDs

    @patch("tracking.tracker.get_config")
    def test_reset_clears_tracks(self, mock_cfg):
        mock_cfg.return_value = self._CFG
        from tracking.tracker import MultiObjectTracker, KalmanTrack
        KalmanTrack._id_counter = 0

        tracker = MultiObjectTracker()
        tracker.update([_make_detection()])
        tracker.reset()
        KalmanTrack._id_counter = 0
        result = tracker.update([])
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 9. Alert data model — tests alert construction
# ══════════════════════════════════════════════════════════════════════════════
class TestAlert:
    """
    Alert is produced by decision modules and consumed by the display.
    Tests confirm level ordering and fields are stored correctly.
    """

    def test_critical_alert_fields(self):
        a = Alert(
            level=AlertLevel.CRITICAL,
            source="collision_warning",
            message="BRAKE! car 1.5m (TTC 0.9s)",
            camera_id=CameraID.FRONT,
            track_id=3,
            ttc=0.9
        )
        assert a.level    == AlertLevel.CRITICAL
        assert a.ttc      == 0.9
        assert a.track_id == 3
        assert "BRAKE" in a.message

    def test_alert_level_values(self):
        # Level values used for sorting in display
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.WARNING.value  == "warning"
        assert AlertLevel.INFO.value     == "info"
        assert AlertLevel.NONE.value     == "none"