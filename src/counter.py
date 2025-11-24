# Line-crossing counting logic

from dataclasses import dataclass, field
from typing import Dict

from src.types import FrameTracks, CountingState


class LineCounter:
    """
    Line-crossing counter (stub version).

    Responsibility:
      - Maintain global counts of how many objects of each class
        have been counted.
      - Later: implement logic to increment counts when tracks cross
        a virtual line.

    For now, this is a placeholder with no real counting logic.
    """

    def __init__(self, line_y: float):
        """
        line_y: relative vertical position of the counting line
                in [0.0, 1.0], where 0 = top, 1 = bottom.
        """
        self.line_y = line_y
        self.state = CountingState()  # total_counts = {}

    def update(self, frame_height: int, frame_tracks: FrameTracks) -> CountingState:
        """
        Update counting state for the current frame.

        Stub behavior:
          - Ignores tracks.
          - Returns current state unchanged.
        """
        # Real logic will go here later (line-crossing detection).
        return self.state
