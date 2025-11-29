from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from src.data_types import FrameDetections, FrameTracks, Track, BoundingBox


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
    Very simple tracker that creates a new track for every detection
    on every frame. Keeps no identity across frames.
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


class SimpleIOUTracker(BaseTracker):
    """
    IOU-based multi-object tracker.

    Logic:
      - Maintains a list of active tracks with stable IDs.
      - For each new frame:
          * Match detections to existing tracks by highest IoU.
          * If IoU >= iou_threshold → update that track.
          * Unmatched detections → create new tracks.
          * Tracks not seen for max_lost_age frames are removed.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost_age = max_lost_age

        self._tracks: List[Track] = []
        self._last_seen: Dict[int, int] = {}   # track_id -> last frame_id
        self._next_id: int = 1

    @staticmethod
    def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """
        Compute Intersection over Union of two boxes in xyxy format.
        """
        x1 = max(box_a.x1, box_b.x1)
        y1 = max(box_a.y1, box_b.y1)
        x2 = min(box_a.x2, box_b.x2)
        y2 = min(box_a.y2, box_b.y2)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, (box_a.x2 - box_a.x1)) * max(0.0, (box_a.y2 - box_a.y1))
        area_b = max(0.0, (box_b.x2 - box_b.x1)) * max(0.0, (box_b.y2 - box_b.y1))

        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0

        return float(inter_area / union)

    def update(self, frame: np.ndarray, detections: FrameDetections) -> FrameTracks:
        frame_id = detections.frame_id
        dets = detections.detections

        # Keep track of which existing tracks already matched a detection
        used_track_ids: set[int] = set()
        new_tracks: List[Track] = []

        # --- Match detections to existing tracks ---
        for det in dets:
            best_iou = 0.0
            best_track_idx = None

            for idx, track in enumerate(self._tracks):
                if track.track_id in used_track_ids:
                    continue

                iou = self._iou(track.box, det.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = idx

            if best_track_idx is not None and best_iou >= self.iou_threshold:
                # Update existing track
                old_track = self._tracks[best_track_idx]
                updated_track = Track(
                    track_id=old_track.track_id,
                    box=BoundingBox(
                        x1=det.box.x1,
                        y1=det.box.y1,
                        x2=det.box.x2,
                        y2=det.box.y2,
                    ),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    score=det.score,
                    age=old_track.age + 1,
                    counted=old_track.counted,
                )
                new_tracks.append(updated_track)
                used_track_ids.add(updated_track.track_id)
                self._last_seen[updated_track.track_id] = frame_id
            else:
                # Create new track
                new_track = Track(
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
                new_tracks.append(new_track)
                self._last_seen[new_track.track_id] = frame_id
                self._next_id += 1

        # --- Keep unmatched old tracks if they are not too old ---
        active_ids = {t.track_id for t in new_tracks}
        for old_track in self._tracks:
            if old_track.track_id in active_ids:
                continue  # already updated

            last_seen = self._last_seen.get(old_track.track_id, frame_id)
            if frame_id - last_seen <= self.max_lost_age:
                # keep this track even though it wasn't matched in this frame
                kept_track = Track(
                    track_id=old_track.track_id,
                    box=old_track.box,
                    class_id=old_track.class_id,
                    class_name=old_track.class_name,
                    score=old_track.score,
                    age=old_track.age + 1,
                    counted=old_track.counted,
                )
                new_tracks.append(kept_track)

        # Update internal state
        self._tracks = new_tracks

        return FrameTracks(frame_id=frame_id, tracks=new_tracks)
