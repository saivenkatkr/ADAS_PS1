"""
perception/yolov8_detector.py
──────────────────────────────
Fixed: uses YOLO's own model.names dict as the ground truth label source.
raw_label is always set to the exact string YOLO returns (e.g. "dog", "cat").
ObjectClass.from_yolo_id() is used for logic (collision, blind spot etc.)
but the display always uses raw_label so labels are always correct.
"""

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from perception.base_detector import BaseDetector, DetectorFactory
from utils.data_models import BoundingBox, CameraID, Detection, ObjectClass


@DetectorFactory.register("yolov8")
class YOLOv8Detector(BaseDetector):

    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "yolov8n.pt")
        self.confidence = config.get("confidence", 0.40)    # slightly lower = catch more
        self.iou        = config.get("iou_threshold", 0.45)
        self.classes    = config.get("classes", None)       # None = all 80 COCO classes
        self._device    = self._resolve_device(config.get("device", "auto"))
        self._names     = {}   # populated after model loads

        logger.info(f"Loading YOLOv8 from '{self.model_path}' on {self._device}")
        self._model = YOLO(self.model_path)
        self._model.to(self._device)

        # Cache the model's own class name dictionary (80 COCO classes)
        self._names = self._model.names   # {0: 'person', 15: 'cat', 16: 'dog', ...}
        logger.info(f"Model has {len(self._names)} classes: {list(self._names.values())[:10]}...")

    def detect(self, frame: np.ndarray, camera_id: CameraID) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
            device=self._device
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])

                # ── CRITICAL FIX ───────────────────────────────────────────
                # raw_label: the exact name from YOLO's own dictionary.
                # This is ALWAYS correct regardless of our ObjectClass enum.
                raw_label = self._names.get(cls_id, f"class_{cls_id}")

                # class_name: our typed enum — used for decision logic only
                # First try matching by name (most reliable), then by ID
                class_name = ObjectClass.from_yolo_name(raw_label)
                if class_name == ObjectClass.UNKNOWN:
                    class_name = ObjectClass.from_yolo_id(cls_id)
                # ──────────────────────────────────────────────────────────

                detections.append(Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                    camera_id=camera_id,
                    raw_label=raw_label      # always the real YOLO label
                ))
        return detections

    def warmup(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, CameraID.FRONT)
        logger.info("[YOLOv8] Warmup complete")

    @property
    def name(self) -> str:
        return f"YOLOv8({self.model_path})"

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            if torch.cuda.is_available():    return "cuda"
            if torch.backends.mps.is_available(): return "mps"
            return "cpu"
        return device_str