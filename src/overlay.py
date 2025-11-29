# Drawing bounding boxes, labels, and counts on frames

import cv2

from src.data_types import FrameTracks, CountingState
from typing import Optional


def draw_tracks_and_counts(
    frame,
    frame_tracks: FrameTracks,
    counts: CountingState,
    line_x: Optional[float] = None,
):
    """
    Draw bounding boxes, track IDs, class labels, and total counts on the frame.

    frame: numpy array (BGR)
    frame_tracks: tracks for this frame
    counts: global CountingState
    line_y: optional float in [0,1] representing the vertical position 
            of the counting line as a percentage of frame height
    """

    h, w = frame.shape[:2]

    # ----- Draw tracks -----
    for track in frame_tracks.tracks:
        x1 = int(track.box.x1)
        y1 = int(track.box.y1)
        x2 = int(track.box.x2)
        y2 = int(track.box.y2)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label: class + track ID
        label = f"{track.class_name}#{track.track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # ----- Draw counts -----
    y = 20
    for cls, count in counts.total_counts.items():
        txt = f"{cls}: {count}"
        cv2.putText(
            frame,
            txt,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25

    # ----- Draw vertical counting line -----
    if line_x is not None:
        line_pixel = int(line_x * w)
        cv2.line(frame, (line_pixel, 0), (line_pixel, h), (0, 0, 255), 2)
