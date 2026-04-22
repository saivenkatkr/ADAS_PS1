"""
perception/depth_estimator.py
Fixes: timm auto-install + isl-org/MiDaS fallback + safe error handling
"""

import subprocess
import sys
import numpy as np
import torch
import cv2
from loguru import logger
from typing import Optional

from utils.data_models import CameraID, DepthMap
from utils.config_loader import get_config


class DepthEstimator:

    def __init__(self):
        cfg = get_config()["perception"]["depth_estimator"]
        self.enabled    = cfg.get("enabled", True)
        self.model_type = cfg.get("model_type", "MiDaS_small")
        self._model     = None
        self._transform = None
        self._device    = "cuda" if torch.cuda.is_available() else "cpu"

        if self.enabled:
            self._load_model()

    def _ensure_timm(self):
        try:
            import timm  # noqa
        except ImportError:
            logger.info("timm not found — installing automatically ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "-q"])
            logger.info("timm installed")

    def _load_model(self):
        logger.info(f"Loading MiDaS ({self.model_type}) on {self._device} ...")
        self._ensure_timm()

        repos = ["isl-org/MiDaS", "intel-isl/MiDaS"]
        for repo in repos:
            try:
                logger.info(f"Trying MiDaS from {repo} ...")
                midas = torch.hub.load(repo, self.model_type, pretrained=True, trust_repo=True)
                midas.eval()
                self._model = midas.to(self._device)

                transforms = torch.hub.load(repo, "transforms", trust_repo=True)
                self._transform = transforms.dpt_transform if "DPT" in self.model_type else transforms.small_transform

                logger.info(f"MiDaS loaded OK from {repo}")
                return
            except Exception as e:
                logger.warning(f"MiDaS failed ({repo}): {e}")

        logger.error("MiDaS unavailable — depth estimation disabled")
        self.enabled = False

    def estimate(self, frame: np.ndarray, camera_id: CameraID) -> DepthMap:
        if not self.enabled or self._model is None:
            return DepthMap(map=None, camera_id=camera_id)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self._device)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
            d = pred.cpu().numpy()
            lo, hi = d.min(), d.max()
            if hi > lo:
                d = (d - lo) / (hi - lo)
            return DepthMap(map=d.astype(np.float32), camera_id=camera_id)
        except Exception as e:
            logger.warning(f"Depth estimate error: {e}")
            return DepthMap(map=None, camera_id=camera_id)

    def get_object_depth(self, depth_map: DepthMap, bbox) -> Optional[float]:
        if depth_map.map is None:
            return None
        roi = depth_map.map[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        return float(np.median(roi)) if roi.size > 0 else None