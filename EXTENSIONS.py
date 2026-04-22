"""
FUTURE EXTENSIONS GUIDE
════════════════════════

HOW TO ADD NEW FEATURES WITHOUT BREAKING EXISTING CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

=============================================================
1. TRAFFIC SIGN DETECTION
=============================================================

Step 1 — Create perception/sign_detector.py:

    from perception.base_detector import BaseDetector, DetectorFactory

    @DetectorFactory.register("sign_detector")
    class TrafficSignDetector(BaseDetector):
        def __init__(self, config):
            # Load pretrained sign model (e.g., YOLOv8 fine-tuned on GTSRB)
            self._model = YOLO(config["model_path"])

        def detect(self, frame, camera_id):
            # Returns List[Detection] with sign classes
            ...

Step 2 — Add sign classes to utils/data_models.py:

    class SignClass(str, Enum):
        STOP        = "stop"
        SPEED_30    = "speed_30"
        GIVE_WAY    = "give_way"
        NO_ENTRY    = "no_entry"

Step 3 — Create decision/sign_logic.py:

    class TrafficSignLogic:
        def process(self, frame_data) -> List[Alert]:
            for sign in frame_data.sign_detections:
                if sign.class_name == SignClass.STOP:
                    return [Alert(level=AlertLevel.WARNING, ...)]

Step 4 — Add 3 lines to main.py _process_frame():

    # In start():
    self.sign_detector = DetectorFactory.create(cfg["perception"]["sign_detector"])
    self.sign_logic    = TrafficSignLogic()

    # In _process_frame():
    frame_data.sign_detections = self.sign_detector.detect(frame, cam_id)
    frame_data.alerts += self.sign_logic.process(frame_data)

Pretrained model options:
  - YOLOv8 fine-tuned on GTSRB (German Traffic Sign Recognition Benchmark)
  - YOLO-NAS pretrained on LISA dataset (US signs)
  - Download: https://universe.roboflow.com/search?q=traffic+signs


=============================================================
2. DRIVER MONITORING SYSTEM (DMS)
=============================================================

Step 1 — Add an interior (driver-facing) camera to config:

    cameras:
      driver:
        source: 4
        resolution: [640, 480]
        fps: 15
        enabled: true

Step 2 — Create perception/driver_monitor.py:

    import mediapipe as mp   # pip install mediapipe

    class DriverMonitor:
        def __init__(self):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        def analyze(self, frame) -> DriverState:
            # Detect: eye aspect ratio (drowsiness)
            #         head pose (distraction direction)
            #         gaze direction (looking away)
            ...

    @dataclass
    class DriverState:
        drowsiness_score: float   # 0-1
        distraction_score: float  # 0-1
        gaze_direction: str       # "forward" | "left" | "right" | "down"
        eyes_closed_sec: float    # duration eyes have been closed

Step 3 — Create decision/driver_alert.py:

    class DriverAlertSystem:
        def process(self, driver_state) -> List[Alert]:
            if driver_state.eyes_closed_sec > 2.0:
                return [Alert(level=AlertLevel.CRITICAL,
                              message="DROWSINESS DETECTED — Wake up!")]
            if driver_state.distraction_score > 0.7:
                return [Alert(level=AlertLevel.WARNING,
                              message="Eyes on road!")]

Key libraries:
  - MediaPipe FaceMesh: landmark-based, runs at 30fps on CPU
  - OpenCV DNN with pretrained facial landmark models
  - dlib 68-point face landmark detector


=============================================================
3. BETTER DEPTH ESTIMATION (STEREO OR METRIC)
=============================================================

Option A — Metric monocular depth with ZoeDepth:

    class ZoeDepthEstimator(DepthEstimator):
        def _load_model(self):
            self._model = torch.hub.load(
                "isl-org/ZoeDepth", "ZoeD_N",
                pretrained=True
            ).to(self._device)
            # ZoeDepth outputs metric depth in meters (not relative!)

    # Change config: depth_estimator.backend: zoedepth
    # No other code changes needed

Option B — Stereo camera (most accurate, requires hardware):

    class StereoDepthEstimator:
        def __init__(self, baseline_m=0.12, focal_px=700):
            self.baseline  = baseline_m
            self.focal     = focal_px
            self.stereo    = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        def estimate(self, left_frame, right_frame) -> DepthMap:
            gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            disp   = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            # True metric depth: Z = (baseline × focal) / disparity
            with np.errstate(divide='ignore'):
                depth_m = (self.baseline * self.focal) / disp
            depth_m[disp <= 0] = 0
            return DepthMap(map=depth_m)

Option C — LiDAR fusion (production):
  - Use ROS2 to fuse camera + LiDAR point cloud
  - Project LiDAR points onto camera image plane
  - Use sparse LiDAR depth to supervise dense monocular estimation


=============================================================
4. NIGHT / ADVERSE WEATHER MODE
=============================================================

    class AdaptivePreprocessor:
        def preprocess(self, frame, conditions="day"):
            if conditions == "night":
                # CLAHE for low-light enhancement
                lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l     = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            elif conditions == "rain":
                # Deraining with pretrained MPRNet or DRD-Net
                frame = self._derain(frame)
            return frame

    # Plug into camera manager's _read_loop() before enqueuing frame


=============================================================
5. REPLACING YOLOV8 WITH ANOTHER MODEL
=============================================================

Example: swap to RT-DETR (transformer-based, no NMS needed)

    # perception/rtdetr_detector.py

    from perception.base_detector import BaseDetector, DetectorFactory

    @DetectorFactory.register("rtdetr")
    class RTDETRDetector(BaseDetector):
        def __init__(self, config):
            from ultralytics import RTDETR
            self._model = RTDETR(config.get("model_path", "rtdetr-l.pt"))

        def detect(self, frame, camera_id):
            results = self._model.predict(frame, conf=0.5, verbose=False)
            # Convert to List[Detection] — same as YOLOv8 format
            ...

        def warmup(self): ...

        @property
        def name(self): return "RT-DETR"

    # Then in config/settings.yaml:
    #   perception.detector.backend: rtdetr
    #   perception.detector.model_path: rtdetr-l.pt


=============================================================
6. SPEED ESTIMATION FROM TRACKS
=============================================================

    class SpeedEstimator:
        \"\"\"Estimates speed of tracked vehicles in km/h.\"\"\"
        def __init__(self, fps=30, pixels_per_meter=50):
            self.fps = fps
            self.ppm = pixels_per_meter
            self._history = {}   # track_id → deque of positions

        def update(self, tracks: List[Track]) -> Dict[int, float]:
            speeds = {}
            for track in tracks:
                cx, cy = track.bbox.center
                if track.track_id not in self._history:
                    self._history[track.track_id] = []
                hist = self._history[track.track_id]
                hist.append((cx, cy))
                if len(hist) > self.fps:
                    hist.pop(0)
                if len(hist) >= 2:
                    dx = hist[-1][0] - hist[0][0]
                    dy = hist[-1][1] - hist[0][1]
                    dist_px  = (dx**2 + dy**2) ** 0.5
                    dist_m   = dist_px / self.ppm
                    elapsed  = len(hist) / self.fps
                    speed_ms = dist_m / elapsed
                    speeds[track.track_id] = speed_ms * 3.6  # → km/h
            return speeds
"""
