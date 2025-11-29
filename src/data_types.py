# Core data structures (boxes, detections, tracks, counts)

from dataclasses import dataclass, field
from typing import List, Dict

# box coordinates
@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in image pixel coordinates.
    (x1, y1) = top-left, (x2, y2) = bottom-right
    """
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Detection:
    """
    Single detection output by the detector for one object.
    """
    box: BoundingBox
    score: float
    class_id: int
    class_name: str

@dataclass
class FrameDetections:
    """
    All detections for a single frame.
    """
    frame_id: int
    detections: List[Detection]

@dataclass
class Track:
    """
    A tracked object across frames.
    """
    track_id: int
    box: BoundingBox
    class_id: int
    class_name: str
    score: float
    age: int            # number of frames this track has been alive
    counted: bool = False  # has this track already been counted
    is_classified: bool = field(default=False)  # store the classification state

@dataclass
class FrameTracks:
    """
    All active tracks for a single frame.
    """
    frame_id: int
    tracks: List[Track]

@dataclass
class CountingState:
    """
    Global counting state across frames.
    total_counts maps class_name -> count.
    """
    total_counts: Dict[str, int] = field(default_factory=dict)
