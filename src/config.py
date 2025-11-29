# all configurations in one place

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class VideoConfig:
    source: Union[int, str] = 0  # 0 for webcam, or path to video file
    frame_width: int = 1500
    frame_height: int = 900
    fps: int = 30

@dataclass
class DetectionConfig:
    model_path: Path = MODELS_DIR / "detector" / "yolo_trashnet.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda"  # or "cuda"

@dataclass
class CountingConfig:
    line_position: float = 0.6  # relative vertical position of virtual line (0-1)
    min_track_age: int = 3      # frames before eligible to count
    max_lost_age: int = 15      # frames to keep lost tracks

@dataclass
class LoggingConfig:
    log_path: Path = DATA_DIR / "logs" / "events.csv"

@dataclass
class PipelineConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
