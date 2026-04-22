"""
demo.py  —  OPTIMIZED VERSION
─────────────────────────────
Speed fixes applied:
  1. MiDaS depth runs every N frames only (configurable)
  2. Frame resize before inference (smaller = faster YOLO)
  3. Frame skipping when pipeline is behind camera
  4. Thread-separated capture vs inference
  5. Depth disabled by default on CPU (biggest speedup)
  6. YOLOv8 imgsz=320 for faster inference on CPU
"""

import argparse
import sys
import time
import threading
from queue import Queue, Empty

import numpy as np
import cv2
from loguru import logger

_DEMO_CONFIG = {
    "system": {"mode": "demo", "target_fps": 30, "show_display": True},
    "cameras": {
        "front": {"source": 0, "resolution": [1280, 720], "fps": 30, "enabled": True},
        "left":  {"source": 0, "resolution": [1280, 720], "fps": 30, "enabled": False},
        "right": {"source": 0, "resolution": [1280, 720], "fps": 30, "enabled": False},
        "rear":  {"source": 0, "resolution": [1280, 720], "fps": 30, "enabled": False},
    },
    "perception": {
        "detector": {
            "backend": "yolov8",
            "model_path": "yolov8n.pt",
            "confidence": 0.45,
            "iou_threshold": 0.45,
            "classes": None,
            "device": "auto"
        },
        "lane_detector": {"backend": "heuristic", "roi_top_ratio": 0.55},
        "depth_estimator": {
            "backend": "midas",
            "model_type": "MiDaS_small",
            "enabled": False      # ← OFF by default on CPU. Set True if you have GPU.
        }
    },
    "tracking": {"backend": "deepsort", "max_age": 30, "min_hits": 2, "iou_threshold": 0.3},
    "decision": {
        "collision":  {"ttc_warning_sec": 3.0, "ttc_critical_sec": 1.5, "min_object_area": 2000},
        "blind_spot": {"zone_x_ratio": 0.25, "min_velocity_kmh": 10.0},
        "parking":    {"critical_distance_cm": 40, "warning_distance_cm": 100, "grid_cells": [10, 6]}
    },
    "output": {
        "display": {"font_scale": 0.6, "overlay_alpha": 0.4},
        "audio":   {"enabled": False, "beep_cooldown_sec": 2.0},
        "logger":  {"enabled": False, "log_dir": "logs/", "record_video": False, "video_dir": "logs/video/"}
    }
}

# ── Performance knobs — tune these for your machine ────────────────────────
INFER_WIDTH      = 416      # resize frame to this width before YOLO (320=fastest, 640=accurate)
DEPTH_EVERY_N    = 10       # run MiDaS every N frames (0 = disabled)
LANE_EVERY_N     = 3        # run lane detection every N frames
DISPLAY_WIDTH    = 960      # resize display window (doesn't affect inference)
CAPTURE_QUEUE_SZ = 2        # max buffered frames (keep small = low latency)
# ──────────────────────────────────────────────────────────────────────────


def run_demo(source=0, enable_depth=False, infer_width=INFER_WIDTH):
    import utils.config_loader as cfg_mod
    _DEMO_CONFIG["cameras"]["front"]["source"] = source
    _DEMO_CONFIG["perception"]["depth_estimator"]["enabled"] = enable_depth
    cfg_mod._config = _DEMO_CONFIG

    from perception.base_detector import DetectorFactory
    import perception.yolov8_detector  # noqa
    from perception.lane_detector import LaneDetector
    from perception.depth_estimator import DepthEstimator
    from tracking.tracker import MultiObjectTracker
    from decision.collision_warning import CollisionWarning
    from output.display import DisplayManager
    from utils.data_models import CameraID, FrameData

    logger.info("=== ADAS Demo (Optimized) ===")
    logger.info(f"Inference width: {infer_width}px | Depth: {'ON every '+str(DEPTH_EVERY_N)+' frames' if enable_depth else 'OFF'}")

    # ── Init modules ────────────────────────────────────────────────────
    detector  = DetectorFactory.create(_DEMO_CONFIG["perception"]["detector"])
    lane_det  = LaneDetector()
    depth_est = DepthEstimator()
    tracker   = MultiObjectTracker()
    collision = CollisionWarning()
    display   = DisplayManager()

    logger.info("Warming up YOLO ...")
    detector.warmup()

    # ── Camera capture thread ────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize capture buffer lag

    frame_queue: Queue = Queue(maxsize=CAPTURE_QUEUE_SZ)
    stop_event = threading.Event()

    def capture_loop():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break
            if frame_queue.full():
                try: frame_queue.get_nowait()   # drop oldest to stay live
                except Empty: pass
            frame_queue.put(frame)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # ── Inference loop ───────────────────────────────────────────────────
    frame_idx   = 0
    fps_times   = []
    cached_lanes = None
    cached_depth = None

    logger.info("Running — press Q to quit")

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue

        t0 = time.monotonic()

        # 1. Resize for faster inference (keep original for display)
        h, w = frame.shape[:2]
        scale = infer_width / w
        small = cv2.resize(frame, (infer_width, int(h * scale)))

        # 2. Detection on small frame
        from utils.data_models import CameraID
        detections_small = detector.detect(small, CameraID.FRONT)

        # 3. Scale bounding boxes back to original frame size
        from utils.data_models import Detection, BoundingBox
        detections = []
        inv = 1.0 / scale
        for d in detections_small:
            b = d.bbox
            detections.append(Detection(
                bbox=BoundingBox(
                    int(b.x1 * inv), int(b.y1 * inv),
                    int(b.x2 * inv), int(b.y2 * inv)
                ),
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                camera_id=d.camera_id,
                raw_label=d.raw_label       # ← carry the correct YOLO label
            ))

        # 4. Lane detection — every N frames only
        if frame_idx % LANE_EVERY_N == 0:
            cached_lanes = lane_det.detect(frame, CameraID.FRONT)

        # 5. Depth — every N frames only (skip entirely if disabled)
        if enable_depth and DEPTH_EVERY_N > 0 and frame_idx % DEPTH_EVERY_N == 0:
            cached_depth = depth_est.estimate(frame, CameraID.FRONT)

        # 6. Tracking
        tracks = tracker.update(detections)

        # 7. Build FrameData and run decisions
        fd = FrameData(
            frame=frame,
            camera_id=CameraID.FRONT,
            timestamp=time.time(),
            frame_idx=frame_idx
        )
        fd.detections = detections
        fd.tracks     = tracks
        fd.lanes      = cached_lanes
        fd.depth      = cached_depth
        fd.alerts    += collision.process(fd)

        # 8. Render
        annotated = display.render(fd)

        # FPS overlay
        fps_times.append(time.monotonic())
        fps_times = [t for t in fps_times if fps_times[-1] - t < 1.0]
        fps = len(fps_times)
        cv2.putText(annotated, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated, f"Infer size: {infer_width}px", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Resize display if needed
        if DISPLAY_WIDTH and annotated.shape[1] != DISPLAY_WIDTH:
            dh = int(annotated.shape[0] * DISPLAY_WIDTH / annotated.shape[1])
            annotated = cv2.resize(annotated, (DISPLAY_WIDTH, dh))

        cv2.imshow("ADAS Demo — press Q to quit", annotated)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if frame_idx % 30 == 0:
            logger.debug(f"Frame {frame_idx} | {elapsed_ms:.1f}ms | FPS {fps} | tracks {len(tracks)}")

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo stopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ADAS Demo — Optimized")
    p.add_argument("--source",  default=0,     help="Camera index or video file")
    p.add_argument("--depth",   action="store_true", help="Enable MiDaS depth (slow on CPU)")
    p.add_argument("--width",   type=int, default=INFER_WIDTH,
                   help=f"Inference frame width (default {INFER_WIDTH}). Lower=faster, 320/416/640")
    args = p.parse_args()

    src = args.source
    try: src = int(src)
    except ValueError: pass

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    run_demo(src, enable_depth=args.depth, infer_width=args.width)