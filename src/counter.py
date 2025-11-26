from typing import Dict

from src.types import FrameTracks, CountingState, Track


class LineCounter:
    """
    Line-crossing counter for a VERTICAL counting line.

    Interpretation:
      - line_position (from config) is horizontal:
          0.0 = left edge, 1.0 = right edge
      - A track is counted when its center X crosses the vertical line
        from LEFT to RIGHT for the first time.

    Behavior:
      - Maintains a global CountingState (class_name -> count).
      - Uses per-track previous center_x to detect crossings.
      - Uses Track.counted flag to avoid double-counting.
    """

    def __init__(self, line_position: float):
        # relative horizontal position of the line in [0, 1]
        self.line_x_rel = line_position
        self.state = CountingState()

        # track_id -> last center x position
        self._prev_center_x: Dict[int, float] = {}

    def update(self, frame_width: int, frame_tracks: FrameTracks) -> CountingState:
        """
        Update counting state for the current frame.

        frame_width: width of the frame in pixels (frame.shape[1])
        frame_tracks: tracks for this frame

        Returns:
            CountingState with updated total_counts.
        """
        line_x = int(self.line_x_rel * frame_width)

        for track in frame_tracks.tracks:
            # current center x of the track's bounding box
            cx = (track.box.x1 + track.box.x2) / 2.0

            # previous center x; if not seen before, initialize to current
            prev_cx = self._prev_center_x.get(track.track_id, cx)

            # detect crossing from left to right through the vertical line:
            # prev < line <= current
            crossed = (prev_cx < line_x) and (cx >= line_x)

            if crossed and not track.counted:
                cls = track.class_name
                self.state.total_counts[cls] = self.state.total_counts.get(cls, 0) + 1
                track.counted = True

            # store current center as previous for next frame
            self._prev_center_x[track.track_id] = cx

        return self.state
