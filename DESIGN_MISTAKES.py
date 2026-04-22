"""
COMMON DESIGN MISTAKES IN ADAS SYSTEMS
═══════════════════════════════════════

Documenting pitfalls so you build it right the first time.


❌ MISTAKE 1 — MONOLITHIC PIPELINE (most common)
──────────────────────────────────────────────────
BAD:
    def process(frame):
        # 500 lines of mixed detection, tracking, decision, display...
        results = yolo.predict(frame)
        for r in results:
            ...  # tracking logic inside detection loop
            if some_condition:
                cv2.putText(frame, ...)  # display mixed with logic

WHY IT'S BAD:
  - Can't unit test individual components
  - Can't swap YOLO without touching display code
  - One bug breaks everything

GOOD: Separate classes, communicate via FrameData (as in this project)


❌ MISTAKE 2 — SYNCHRONOUS CAMERA READS IN THE MAIN THREAD
────────────────────────────────────────────────────────────
BAD:
    while True:
        ret, frame = cap.read()       # blocks main thread
        process_front(frame)
        ret, frame = cap2.read()      # blocks again
        process_rear(frame)

WHY IT'S BAD:
  - 4 cameras × 33ms/frame = 132ms latency minimum
  - GPU is idle during camera reads

GOOD: Each camera in its own daemon thread (see cameras/camera_manager.py)


❌ MISTAKE 3 — RUNNING DEPTH ON EVERY FRAME
─────────────────────────────────────────────
BAD:
    frame_data.depth = depth_estimator.estimate(frame)  # every frame

WHY IT'S BAD:
  - MiDaS takes 15-50ms/frame — kills your FPS

GOOD:
    if frame_data.frame_idx % 3 == 0:   # run every 3rd frame
        frame_data.depth = depth_estimator.estimate(frame)
    # Or: run depth on a separate lower-priority thread


❌ MISTAKE 4 — HARD-CODING THRESHOLDS IN LOGIC
────────────────────────────────────────────────
BAD:
    if distance < 5.0:        # hard-coded in decision module
        alert(...)

WHY IT'S BAD:
  - Tuning requires code changes + redeployment
  - Different vehicles need different thresholds

GOOD: All thresholds in config/settings.yaml, loaded via get_config()


❌ MISTAKE 5 — ONE TRACKER FOR ALL CAMERAS
────────────────────────────────────────────
BAD:
    self.tracker = MultiObjectTracker()   # shared across all cameras

WHY IT'S BAD:
  - Track IDs collide between cameras
  - Left camera track #5 ≠ front camera track #5 but treated as same

GOOD: One tracker instance per camera (see main.py self.trackers dict)


❌ MISTAKE 6 — BLOCKING ON GPU INFERENCE IN DISPLAY LOOP
─────────────────────────────────────────────────────────
BAD:
    result = model.predict(frame)     # 20ms GPU call
    cv2.imshow("ADAS", frame)         # only shows after predict
    cv2.waitKey(1)

WHY IT'S BAD:
  - Display freezes during inference
  - User sees dropped frames and jerky video

GOOD: Use producer-consumer with a render queue, or use cv2.waitKey(1)
      non-blocking and only update display when new annotated frame ready


❌ MISTAKE 7 — TRUSTING DISTANCE FROM BBOX SIZE ALONE
───────────────────────────────────────────────────────
BAD:
    distance = 1000 / bbox.height    # oversimplified

WHY IT'S BAD:
  - Assumes constant object size (cars are not all the same height)
  - Very different at wide angles
  - Near/far estimation from size alone is ±50% error at best

GOOD:
  - Calibrate focal length per camera
  - Use known object heights per class (see collision_warning.py)
  - For production: LiDAR or stereo camera for ground-truth distance


❌ MISTAKE 8 — RUNNING ALL MODULES ON ALL CAMERAS
──────────────────────────────────────────────────
BAD:
    lane_info   = lane_detector.detect(rear_frame)    # meaningless
    blind_spot  = bsd.process(front_frame_data)       # wrong camera

GOOD: Each decision module explicitly checks camera_id (see each module):
    if frame_data.camera_id != CameraID.FRONT:
        return []


❌ MISTAKE 9 — NO WARMUP → FIRST FRAME LATENCY SPIKE
──────────────────────────────────────────────────────
BAD:
    model = YOLO("yolov8n.pt")
    # First inference: 500ms+ (model loads into GPU)
    # User sees false "slow" frame

GOOD: Always call detector.warmup() before starting camera streams
      (see main.py start() method)


❌ MISTAKE 10 — MUTABLE DEFAULT ARGUMENTS IN DATACLASSES
──────────────────────────────────────────────────────────
BAD (Python gotcha):
    @dataclass
    class FrameData:
        alerts: List[Alert] = []   # WRONG — shared across all instances!

WHY IT'S BAD:
  - All FrameData instances share the same alerts list
  - Alerts accumulate across frames → silent corruption

GOOD:
    from dataclasses import field
    alerts: List[Alert] = field(default_factory=list)   # ✓
"""
