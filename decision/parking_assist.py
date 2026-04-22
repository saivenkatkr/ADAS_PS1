"""
decision/parking_assist.py
───────────────────────────
Reverse parking assistance using the REAR camera.

Features:
  1. Detects obstacles behind vehicle
  2. Estimates distance to nearest obstacle
  3. Generates proximity grid (like a parking sensor display)
  4. Issues warnings based on distance thresholds

Distance estimation for parking (short range, <3m):
  Uses bounding box area as a proxy when depth map is unavailable.
  With MiDaS depth: uses actual depth map values for better accuracy.
"""

from typing import List, Tuple, Optional
import numpy as np

from utils.data_models import (
    Alert, AlertLevel, CameraID, DepthMap, FrameData, Track
)
from utils.config_loader import get_config


class ParkingAssist:
    """
    Processes REAR camera frames for parking assistance.
    """

    def __init__(self):
        cfg = get_config()["decision"]["parking"]
        self.critical_cm = cfg.get("critical_distance_cm", 40)
        self.warning_cm  = cfg.get("warning_distance_cm",  100)
        grid_cfg         = cfg.get("grid_cells", [10, 6])
        self.grid_cols   = grid_cfg[0]
        self.grid_rows   = grid_cfg[1]

    def process(self, frame_data: FrameData) -> List[Alert]:
        if frame_data.camera_id != CameraID.REAR:
            return []

        alerts = []
        nearest_dist = self._get_nearest_obstacle(frame_data)

        if nearest_dist is None:
            return []

        # Convert relative depth to approximate cm
        # (0.0 = very close, needs calibration for real metric values)
        proximity = self._depth_to_proximity_cm(nearest_dist)

        if proximity < self.critical_cm:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                source="parking_assist",
                message=f"STOP! Obstacle {proximity:.0f}cm",
                camera_id=CameraID.REAR
            ))
        elif proximity < self.warning_cm:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                source="parking_assist",
                message=f"Obstacle {proximity:.0f}cm behind",
                camera_id=CameraID.REAR
            ))

        return alerts

    def _get_nearest_obstacle(self, frame_data: FrameData) -> Optional[float]:
        """
        Returns the minimum depth value in the lower portion of the rear frame.
        Uses depth map if available, falls back to bounding box heuristic.
        """
        if frame_data.depth and frame_data.depth.map is not None:
            depth_map = frame_data.depth.map
            h = depth_map.shape[0]
            lower_roi = depth_map[h // 2:, :]        # bottom half of frame
            # In MiDaS: lower value = CLOSER (inverted)
            return float(lower_roi.min())

        # Fallback: use largest bounding box = closest object
        if frame_data.tracks:
            largest = max(frame_data.tracks, key=lambda t: t.bbox.area)
            # Normalize area to 0–1 range (bigger = closer)
            frame_area = frame_data.frame.shape[0] * frame_data.frame.shape[1]
            area_ratio = largest.bbox.area / frame_area
            return 1.0 - area_ratio    # invert: large bbox = small depth value

        return None

    def _depth_to_proximity_cm(self, raw_depth: float) -> float:
        """
        Rough conversion from MiDaS relative depth to centimetres.
        IMPORTANT: Calibrate this for your camera + environment.
        
        raw_depth ~0.0 → very close (~20cm)
        raw_depth ~0.5 → medium (~100cm)
        raw_depth ~1.0 → far (>300cm)
        """
        # Linear approximation — replace with proper calibration curve
        approx_cm = raw_depth * 300.0 + 20.0
        return max(10.0, approx_cm)

    def generate_proximity_grid(self, frame_data: FrameData) -> np.ndarray:
        """
        Returns a (rows × cols) float array where each cell = threat level 0–1.
        Used by the HUD to draw the parking sensor grid overlay.
        """
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        if frame_data.depth and frame_data.depth.map is not None:
            h, w = frame_data.depth.map.shape
            cell_h = h // self.grid_rows
            cell_w = w // self.grid_cols

            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    y0, y1 = r * cell_h, (r + 1) * cell_h
                    x0, x1 = c * cell_w, (c + 1) * cell_w
                    cell = frame_data.depth.map[y0:y1, x0:x1]
                    # Low depth value = close = high threat
                    grid[r, c] = 1.0 - float(cell.mean())

        return grid
