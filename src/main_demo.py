# End-to-end conveyor demo script

import argparse
import cv2
import numpy as np
import torch

from src.config import PipelineConfig
from src.detector import YoloDetector
from src.tracker import SimpleIOUTracker
from src.counter import LineCounter, CountingState
from src.overlay import draw_tracks_and_counts


from src.integration_clients import (
    BaseEmbeddingClient,
    BaseAnomalyClient,
    BaseFewShotClient,
)
from src.b_models_impl import (
    MyEmbeddingClient,
    MyAnomalyClient,
    MyFewShotClient,
)

# --- Initialize ML Clients ---
CLASSIFIER_MODEL = 'models/classifier/simple_cnn.pth'
CLASSES_JSON = 'models/classifier/classes.json'
ANOMALY_MODEL = 'models/anomaly/lof_scorer.pkl' 
FEWSHOT_PROTOTYPES = 'models/fewshot/prototypes.pkl'


def _classify_trashnet(crop: np.ndarray, client: MyEmbeddingClient) -> str:
    """
    Classifies a crop into one of the 6 TrashNet classes using the ResNet model.
    """
    
    with torch.no_grad():
        # 1. Preprocess and tensorize the crop
        x = client._preprocess(crop) 
        
        # 2. Run the full forward pass (to get logits from the classifier head)
        logits = client.model(x) 
        
        # 3. Get the predicted class index
        _, pred_idx = torch.max(logits, 1)
        
        # 4. Convert index back to class name
        return client.idx_to_class[str(pred_idx.item())]

def _crop_from_box(frame: np.ndarray, box) -> np.ndarray:
    """
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

    # Detection components
    detector = YoloDetector(config.detection)
    tracker = SimpleIOUTracker(iou_threshold=0.3, max_lost_age=15)
    counter = LineCounter(config.counting.line_position)

    # Classification components
    embedding_client: BaseEmbeddingClient = MyEmbeddingClient(
        model_path=CLASSIFIER_MODEL,
        classes_path=CLASSES_JSON
    )
    anomaly_client: BaseAnomalyClient = MyAnomalyClient(
        model_path=ANOMALY_MODEL,
        # NOTE: Adjust this threshold based on your IsolationForest training results!
        threshold=-0.01 
    ) 
    fewshot_client: BaseFewShotClient = MyFewShotClient(
        prototypes_path=FEWSHOT_PROTOTYPES,
        sim_threshold=0.7 # Adjust this based on confidence testing
    )

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

        # 3) Specialized Classification Logic
        for track in frame_tracks.tracks:

            # Only run the expensive classification if the object is NEW (age < 5) or if it hasn't been successfully classified yet.
            if track.is_classified:
                continue

            # 1. Crop the object using the track's latest box
            crop = _crop_from_box(frame, track.box)
            
            # 2. Skip if the crop is invalid (empty)
            if crop.size == 0:
                continue
                
            # --- Execute ML Pipeline ---
            try:
                # Extract embedding
                emb = embedding_client.embed_crop(crop)

                # Anomaly Check
                if anomaly_client.is_anomalous(emb):
                    fs_result = fewshot_client.classify(emb)
                    
                    if fs_result.is_confident:
                        # Confident Rare Classification
                        track.class_name = f"RARE: {fs_result.label}"
                        track.is_classified = True # Classification is complete
                    else:
                        # Unclassified Anomaly
                        track.class_name = "UNKNOWN ANOMALY"
                        # Keep is_classified = False so it can be re-evaluated later if needed
                
                else:
                    # Standard TrashNet Classification
                    track.class_name = _classify_trashnet(crop, embedding_client)
                    track.is_classified = True # Classification is complete
                    
            except Exception as e:
                # Handle cases where model inference fails (e.g., bad crop, file error)
                print(f"Classification failed for track {track.track_id}: {e}")
                track.class_name = "Error"
                # We can try again later if is_classified is left False

        # 4) Counting
        counts: CountingState = counter.update(frame.shape[1], frame_tracks)

        # 5) Visualization
        draw_tracks_and_counts(
            frame,
            frame_tracks,
            counts,
            line_x=config.counting.line_position,
        )

        cv2.imshow("Smart Waste Conveyor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()


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
