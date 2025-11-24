import numpy as np
from src.detector import YoloDetector
from src.config import DetectionConfig

cfg = DetectionConfig()
detector = YoloDetector(cfg)

frame = np.zeros((480, 640, 3), dtype=np.uint8)
out = detector.detect(frame, frame_id=1)
print(out)
