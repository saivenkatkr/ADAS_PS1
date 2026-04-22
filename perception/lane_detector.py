"""
perception/lane_detector.py
────────────────────────────
Lane detection module.

Backends:
  heuristic → OpenCV Canny + Hough lines (no model needed, fast, robust on highways)
  clrnet    → plug in CLRNet pretrained weights for curved lanes (TODO)

The heuristic approach works well for:
  - Straight and mildly curved roads
  - Good lighting conditions
  - Clear lane markings

For complex scenarios (night, rain, sharp curves) → swap to DL backend.
"""

import cv2
import numpy as np
from typing import Optional
from loguru import logger

from utils.data_models import CameraID, LaneInfo
from utils.config_loader import get_config


class LaneDetector:
    """
    Detects left and right lane lines from a front camera frame.

    Usage:
        detector = LaneDetector()
        lane_info = detector.detect(frame, CameraID.FRONT)
    """

    def __init__(self):
        cfg = get_config()["perception"]["lane_detector"]
        self.backend = cfg.get("backend", "heuristic")
        self.roi_top = cfg.get("roi_top_ratio", 0.55)
        logger.info(f"Lane detector backend: {self.backend}")

    def detect(self, frame: np.ndarray, camera_id: CameraID) -> LaneInfo:
        if self.backend == "heuristic":
            return self._detect_heuristic(frame, camera_id)
        else:
            logger.warning(f"Backend '{self.backend}' not implemented — using heuristic")
            return self._detect_heuristic(frame, camera_id)

    def _detect_heuristic(self, frame: np.ndarray, camera_id: CameraID) -> LaneInfo:
        h, w = frame.shape[:2]
        roi_y = int(h * self.roi_top)

        # 1. Preprocess
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges   = cv2.Canny(blurred, 50, 150)

        # 2. Region of Interest (trapezoid mask)
        mask = np.zeros_like(edges)
        roi_pts = np.array([[
            (int(w * 0.0), h),
            (int(w * 0.45), roi_y),
            (int(w * 0.55), roi_y),
            (int(w * 1.0), h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 3. Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1, theta=np.pi / 180,
            threshold=40,
            minLineLength=80,
            maxLineGap=100
        )

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue  # vertical — skip
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:         # negative slope = left lane
                    left_lines.append(line[0])
                elif slope > 0.5:        # positive slope = right lane
                    right_lines.append(line[0])

        left_line  = self._average_lines(left_lines,  h, roi_y)
        right_line = self._average_lines(right_lines, h, roi_y)

        # 4. Center offset
        center_offset = self._compute_center_offset(left_line, right_line, w)

        departure = abs(center_offset) > w * 0.08   # >8% of width = warning

        return LaneInfo(
            left_line=left_line,
            right_line=right_line,
            center_offset_px=center_offset,
            departure_warning=departure,
            camera_id=camera_id
        )

    def _average_lines(self, lines, frame_h, roi_y) -> Optional[np.ndarray]:
        """Fit a single representative line through all detected line segments."""
        if not lines:
            return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        poly = np.polyfit(y_coords, x_coords, 1)  # x = f(y)
        y1 = frame_h
        y2 = roi_y
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        return np.array([x1, y1, x2, y2])

    def _compute_center_offset(self, left, right, frame_w) -> float:
        """Positive = car is right of center lane."""
        car_center = frame_w // 2
        if left is not None and right is not None:
            left_x  = (left[0]  + left[2])  // 2
            right_x = (right[0] + right[2]) // 2
            lane_center = (left_x + right_x) // 2
            return float(car_center - lane_center)
        return 0.0
