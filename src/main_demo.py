# End-to-end conveyor demo script

import argparse
import cv2
import numpy as np

from src.config import PipelineConfig
from src.detector import YoloDetector
from src.tracker import DummyTracker    # later: replace with real tracker
from src.counter import LineCounter, CountingState
from src.overlay import draw_tracks_and_counts

# If Person B follows your interfaces, you will later add:
# from src.integration_clients import (
#     BaseEmbeddingClient,
#     BaseAnomalyClient,
#     BaseFewShotClient,
# )
# from src.b_models_impl import (
#     MyEmbeddingClient,
#     MyAnomalyClient,
#     MyFewShotClient,
# )


def run_demo(config: PipelineConfig, video_source=None) -> None:
    """
    End-to-end demo:
      frame -> detector -> tracker -> counter -> overlay -> display
    Currently uses DummyDetector and DummyTracker.
    """

    if video_source is None:
        video_source = config.video.source

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    # Person A components
    detector = YoloDetector(config.detection)
    tracker = DummyTracker()
    counter = LineCounter(config.counting.line_position)

    # Person B components will be instantiated here later, e.g.:
    # embedding_client: BaseEmbeddingClient = MyEmbeddingClient(model_path="...")
    # anomaly_client: BaseAnomalyClient = MyAnomalyClient(model_path="...")
    # fewshot_client: BaseFewShotClient = MyFewShotClient(prototypes_path="...")

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Resize to configured resolution
        frame = cv2.resize(
            frame,
            (config.video.frame_width, config.video.frame_height),
        )

        # 1) Detection
        frame_detections = detector.detect(frame, frame_id)

        # 2) Tracking
        frame_tracks = tracker.update(frame, frame_detections)

        # 3) Person B logic would go here later:
        # for track in frame_tracks.tracks:
        #     crop = _crop_from_box(frame, track.box)
        #     emb = embedding_client.embed_crop(crop)
        #
        #     if anomaly_client.is_anomalous(emb):
        #         fs_result = fewshot_client.classify(emb)
        #         if fs_result.is_confident:
        #             track.class_name = fs_result.label
        #         else:
        #             track.class_name = "Unknown"
        #     else:
        #         # keep detector label or let Person B override
        #         pass

        # 4) Counting
        counts: CountingState = counter.update(frame.shape[0], frame_tracks)

        # 5) Visualization
        draw_tracks_and_counts(
            frame,
            frame_tracks,
            counts,
            line_y=config.counting.line_position,
        )

        cv2.imshow("Smart Waste Conveyor (stub)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()


def _crop_from_box(frame: np.ndarray, box) -> np.ndarray:
    """
    Utility for Person B integration.
    Safely crops the region corresponding to a bounding box.
    """
    x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return frame[0:1, 0:1, :]

    return frame[y1:y2, x1:x2, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Waste Conveyor Demo (Person A pipeline)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file path or camera index (e.g. 0 for default webcam)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()

    if args.video is None:
        video_source = cfg.video.source
    else:
        # If argument is a digit, treat it as camera index; else as path
        if args.video.isdigit():
            video_source = int(args.video)
        else:
            video_source = args.video

    run_demo(cfg, video_source=video_source)
