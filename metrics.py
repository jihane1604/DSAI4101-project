import json
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
YOLO_DATA_CONFIG = Path("data/data.yaml") 
CUSTOM_YOLO_WEIGHTS = Path("models/detector/yolo_trashnet.pt") 
FALLBACK_YOLO_MODEL = "yolov8n.pt" 

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_detection_metrics():
    """
    Calculates Mean Average Precision (mAP) and saves the results.
    """
    
    print("--- Running YOLO Detector Validation ---")
    
    # 1. Load the detector model
    if CUSTOM_YOLO_WEIGHTS.is_file():
        model_path = str(CUSTOM_YOLO_WEIGHTS)
    else:
        model_path = FALLBACK_YOLO_MODEL
        print(f"Warning: Using fallback model {model_path} as custom weights not found.")
        
    model = YOLO(model_path)
    
    # 2. Run the validation function
    # model.val() calculates mAP using two primary IoU standards:
    # mAP@50 and mAP@50-95
    results = model.val(
        data=str(YOLO_DATA_CONFIG),
        imgsz=640,
        split='val',
        save_json=False,
        conf=0.25, # Confidence threshold for initial True/False Positive assignment
        # The IoU threshold (0.7) set here is for Non-Maximum Suppression (NMS), 
        # NOT the one used for the final mAP calculation.
        iou=0.7 
    )
    
    # 3. Extract and format metrics
    # The IoU thresholds are implicit in the metric names (mAP50 and mAP50-95)
    metrics = {
        "model_used": model_path,
        "mAP_score_explanation": "mAP is the average precision across all classes, based on specific IoU thresholds.",
        
        # Mean Average Precision at IoU=0.5
        "mAP@50": {
            "value": round(results.results_dict["metrics/mAP50(B)"], 4),
            "iou_threshold": 0.5,
            "description": "Measures performance when a detection is considered True Positive if IoU >= 50%. (Less strict localization)."
        },
        
        # Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
        "mAP@50-95": {
            "value": round(results.results_dict["metrics/mAP50-95(B)"], 4),
            "iou_thresholds": "0.5 to 0.95 (in steps of 0.05)",
            "description": "The standard metric for accurate localization. It averages the mAP across 10 IoU thresholds (0.5, 0.55, 0.6, ..., 0.95)."
        }
    }

    # 4. Save metrics to JSON
    metrics_path = METRICS_DIR / "yolo_detection_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nDetection Metrics Saved to: {metrics_path}")
    print(json.dumps(metrics, indent=4))
    
    return metrics

if __name__ == "__main__":
    calculate_detection_metrics()