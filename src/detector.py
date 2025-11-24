# Detection interface + YOLO implementation

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.types import Detection, FrameDetections, BoundingBox


class BaseDetector(ABC):
    """
    Abstract interface for all detectors.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int) -> FrameDetections:
        """
        Run detection on a single frame.
        Must return FrameDetections.
        """
        raise NotImplementedError


class DummyDetector(BaseDetector):
    """
    Placeholder detector.
    Returns no detections.
    Lets you build and test the pipeline before integrating YOLO.
    """

    def detect(self, frame: np.ndarray, frame_id: int) -> FrameDetections:
        return FrameDetections(frame_id=frame_id, detections=[])
