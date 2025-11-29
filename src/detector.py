from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

import numpy as np

from ultralytics import YOLO

from src.data_types import Detection, FrameDetections, BoundingBox
from src.config import DetectionConfig, MODELS_DIR


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


class YoloDetector(BaseDetector):
    """
    YOLOv8-based detector using the ultralytics package.

    Behavior:
      - If a custom weights file exists at DetectionConfig.model_path, use it.
      - Otherwise, fall back to a pretrained 'yolov8n.pt' model (COCO classes)
        so you can test the pipeline before you train your own detector.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config

        weights_path: Path = Path(self.config.model_path)

        if weights_path.is_file():
            # Use your trained weights (e.g., on TrashNet-style data)
            self.model = YOLO(str(weights_path))
        else:
            # Fallback: small pretrained model for quick testing
            # This will download yolov8n.pt on first use.
            self.model = YOLO("yolov8n.pt")

        # YOLO model has a .names dict: class_id -> class_name
        # Convert to a simple list for fast lookup.
        self.class_names = self.model.names  # usually dict[int, str]

    def detect(self, frame: np.ndarray, frame_id: int) -> FrameDetections:
        """
        Run YOLO detection on a single BGR frame.
        Returns FrameDetections with BoundingBox + class info filtered by confidence.
        """

        # Run inference. ultralytics YOLO accepts numpy arrays directly.
        # We request a single result (index 0).
        results = self.model(
            frame,
            # Pass configuration settings directly to the YOLO model
            conf=0.01, # Use the minimum score defined in your config
            iou=0.7,                              # Recommended NMS IoU threshold: 0.45 to 0.70
            verbose=False,
            # also limit the maximum number of detections
            max_det=10,
        )[0]

        detections: List[Detection] = []

        if results.boxes is None:
            return FrameDetections(frame_id=frame_id, detections=detections)

        # Each box in results.boxes has xyxy, conf, cls
        for box in results.boxes:
            # Confidence
            score = float(box.conf[0].item())
            if score < self.config.confidence_threshold:
                continue

            # Class id and name
            class_id = int(box.cls[0].item())
            # model.names may be dict or list; both support [] lookup
            class_name = str(self.class_names[class_id])

            # Bounding box coordinates in absolute pixels (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det = Detection(
                box=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                ),
                score=score,
                class_id=class_id,
                class_name=class_name,
            )
            detections.append(det)

        return FrameDetections(frame_id=frame_id, detections=detections)
