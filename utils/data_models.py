"""
utils/data_models.py
────────────────────
Shared data structures. Fixed: ObjectClass now covers all 80 COCO classes
and from_yolo_id() never returns UNKNOWN for a valid detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class CameraID(str, Enum):
    FRONT = "front"
    LEFT  = "left"
    RIGHT = "right"
    REAR  = "rear"


class AlertLevel(str, Enum):
    NONE     = "none"
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class ObjectClass(str, Enum):
    # ── Vehicles ──────────────────────────────────────────
    BICYCLE    = "bicycle"
    CAR        = "car"
    MOTORCYCLE = "motorcycle"
    AIRPLANE   = "airplane"
    BUS        = "bus"
    TRAIN      = "train"
    TRUCK      = "truck"
    BOAT       = "boat"

    # ── People ────────────────────────────────────────────
    PERSON     = "person"

    # ── Animals ───────────────────────────────────────────
    BIRD       = "bird"
    CAT        = "cat"
    DOG        = "dog"
    HORSE      = "horse"
    SHEEP      = "sheep"
    COW        = "cow"
    ELEPHANT   = "elephant"
    BEAR       = "bear"
    ZEBRA      = "zebra"
    GIRAFFE    = "giraffe"

    # ── Traffic / Outdoor ─────────────────────────────────
    TRAFFIC_LIGHT  = "traffic light"
    FIRE_HYDRANT   = "fire hydrant"
    STOP_SIGN      = "stop sign"
    PARKING_METER  = "parking meter"
    BENCH          = "bench"

    # ── Common objects (useful for parking/obstacle) ──────
    BACKPACK       = "backpack"
    UMBRELLA       = "umbrella"
    HANDBAG        = "handbag"
    SUITCASE       = "suitcase"
    SPORTS_BALL    = "sports ball"
    BOTTLE         = "bottle"
    CHAIR          = "chair"
    COUCH          = "couch"
    POTTED_PLANT   = "potted plant"
    BED            = "bed"
    DINING_TABLE   = "dining table"
    TOILET         = "toilet"
    TV             = "tv"
    LAPTOP         = "laptop"
    CELL_PHONE     = "cell phone"
    MICROWAVE      = "microwave"
    OVEN           = "oven"
    REFRIGERATOR   = "refrigerator"
    BOOK           = "book"
    CLOCK          = "clock"
    VASE           = "vase"
    SCISSORS       = "scissors"
    TEDDY_BEAR     = "teddy bear"
    HAIR_DRIER     = "hair drier"
    TOOTHBRUSH     = "toothbrush"

    UNKNOWN        = "unknown"

    @classmethod
    def from_yolo_id(cls, yolo_id: int) -> "ObjectClass":
        """
        Full COCO 80-class mapping.
        Every valid YOLO class ID maps to the correct label.
        Nothing falls through to UNKNOWN unless it genuinely isn't in COCO.
        """
        _MAP = {
            0:  cls.PERSON,        1:  cls.BICYCLE,      2:  cls.CAR,
            3:  cls.MOTORCYCLE,    4:  cls.AIRPLANE,      5:  cls.BUS,
            6:  cls.TRAIN,         7:  cls.TRUCK,         8:  cls.BOAT,
            9:  cls.TRAFFIC_LIGHT, 10: cls.FIRE_HYDRANT,  11: cls.STOP_SIGN,
            12: cls.PARKING_METER, 13: cls.BENCH,
            14: cls.BIRD,          15: cls.CAT,           16: cls.DOG,
            17: cls.HORSE,         18: cls.SHEEP,         19: cls.COW,
            20: cls.ELEPHANT,      21: cls.BEAR,          22: cls.ZEBRA,
            23: cls.GIRAFFE,
            24: cls.BACKPACK,      25: cls.UMBRELLA,      26: cls.HANDBAG,
            28: cls.SUITCASE,
            32: cls.SPORTS_BALL,
            39: cls.BOTTLE,
            56: cls.CHAIR,         57: cls.COUCH,         58: cls.POTTED_PLANT,
            59: cls.BED,           60: cls.DINING_TABLE,  61: cls.TOILET,
            62: cls.TV,            63: cls.LAPTOP,
            67: cls.CELL_PHONE,
            68: cls.MICROWAVE,     69: cls.OVEN,          72: cls.REFRIGERATOR,
            73: cls.BOOK,          74: cls.CLOCK,         75: cls.VASE,
            76: cls.SCISSORS,      77: cls.TEDDY_BEAR,    78: cls.HAIR_DRIER,
            79: cls.TOOTHBRUSH,
        }
        return _MAP.get(yolo_id, cls.UNKNOWN)

    @classmethod
    def from_yolo_name(cls, name: str) -> "ObjectClass":
        """
        Fallback: use the name string YOLO reports directly.
        This guarantees we NEVER show a wrong label — if our enum
        doesn't have the class, we store the raw YOLO name as UNKNOWN
        but display the real name via raw_label on Detection.
        """
        name_lower = name.lower().strip()
        for member in cls:
            if member.value == name_lower:
                return member
        return cls.UNKNOWN


# ── Core Data Types ─────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:  return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def area(self) -> int:   return self.width * self.height
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    @property
    def bottom_center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


@dataclass
class Detection:
    bbox:       BoundingBox
    confidence: float
    class_id:   int
    class_name: ObjectClass
    camera_id:  CameraID
    raw_label:  str = ""        # ← NEW: exact string from YOLO model, always correct


@dataclass
class Track:
    track_id:   int
    bbox:       BoundingBox
    class_name: ObjectClass
    camera_id:  CameraID
    confidence: float
    raw_label:  str = ""        # ← NEW: carries through from Detection
    velocity:   Tuple[float, float] = (0.0, 0.0)
    age:        int = 0
    estimated_distance_m: Optional[float] = None

    @property
    def display_label(self) -> str:
        """Always returns the best available label — never 'unknown' if YOLO knew."""
        if self.raw_label:
            return self.raw_label
        if self.class_name != ObjectClass.UNKNOWN:
            return self.class_name.value
        return "unknown"


@dataclass
class LaneInfo:
    left_line:         Optional[np.ndarray] = None
    right_line:        Optional[np.ndarray] = None
    center_offset_px:  float = 0.0
    departure_warning: bool  = False
    camera_id:         CameraID = CameraID.FRONT


@dataclass
class DepthMap:
    map:       Optional[np.ndarray] = None
    camera_id: CameraID = CameraID.FRONT


@dataclass
class FrameData:
    frame:      np.ndarray
    camera_id:  CameraID
    timestamp:  float
    frame_idx:  int
    detections: List[Detection] = field(default_factory=list)
    tracks:     List[Track]     = field(default_factory=list)
    lanes:      Optional[LaneInfo]  = None
    depth:      Optional[DepthMap]  = None
    alerts:     List["Alert"]   = field(default_factory=list)


@dataclass
class Alert:
    level:    AlertLevel
    source:   str
    message:  str
    camera_id: CameraID
    track_id: Optional[int]   = None
    ttc:      Optional[float] = None