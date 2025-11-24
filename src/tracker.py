# Tracking interface + simple tracker

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.types import FrameDetections, FrameTracks, Track, BoundingBox


class BaseTracker(ABC):
    """
    Abstract interface for all trackers.
    """

    @abstractmethod
    def update(self, frame: np.ndarray, detections: FrameDetections) -> FrameTracks:
        """
        Update tracker with detections for current frame.
        Must return FrameTracks.
        """
        raise NotImplementedError


class DummyTracker(BaseTracker):
    """
    Placeholder tracker.
    For now:
      - creates one Track per Detection
      - assigns a new track_id each call
      - does NOT maintain IDs across frames
    This is enough to wire the pipeline and test overlay and counting structure.
    """

    def __init__(self):
        self._next_id: int = 1

    def update(self, frame: np.ndarray, detections: FrameDetections) -> FrameTracks:
        tracks: List[Track] = []

        for det in detections.detections:
            track = Track(
                track_id=self._next_id,
                box=BoundingBox(
                    x1=det.box.x1,
                    y1=det.box.y1,
                    x2=det.box.x2,
                    y2=det.box.y2,
                ),
                class_id=det.class_id,
                class_name=det.class_name,
                score=det.score,
                age=0,
                counted=False,
            )
            tracks.append(track)
            self._next_id += 1

        return FrameTracks(frame_id=detections.frame_id, tracks=tracks)
