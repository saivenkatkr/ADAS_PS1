"""
perception/base_detector.py
────────────────────────────
Abstract base class for ALL object detectors.

The key to flexibility:
  All detector implementations MUST follow this interface.
  The pipeline only ever calls detect(frame) → never touches model internals.
  Swapping YOLOv8 → RTDETRv2 = change config + new class, zero pipeline changes.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from utils.data_models import CameraID, Detection


class BaseDetector(ABC):
    """
    Contract every detector must fulfill.
    
    To add a new detector:
    1. Create NewDetector(BaseDetector) in perception/
    2. Implement detect() returning List[Detection]
    3. Register it in DetectorFactory below
    """

    @abstractmethod
    def detect(self, frame: np.ndarray, camera_id: CameraID) -> List[Detection]:
        """
        Run inference on one frame.
        
        Args:
            frame: BGR numpy array (H, W, 3)
            camera_id: which camera this frame came from
        Returns:
            List of Detection objects (may be empty)
        """
        ...

    @abstractmethod
    def warmup(self):
        """Run one dummy inference to load weights into GPU memory."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name for logging."""
        ...


class DetectorFactory:
    """
    Registry pattern — maps config string → detector class.
    No if/else chains in the pipeline; just look up the right class.
    """

    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a detector class."""
        def decorator(detector_cls):
            cls._registry[name] = detector_cls
            return detector_cls
        return decorator

    @classmethod
    def create(cls, config: dict) -> BaseDetector:
        backend = config.get("backend", "yolov8")
        if backend not in cls._registry:
            raise ValueError(
                f"Unknown detector backend '{backend}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[backend](config)
