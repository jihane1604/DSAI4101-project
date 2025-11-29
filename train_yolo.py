import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO
import argparse
from pathlib import Path

# Define paths relative to the script execution location
DATA_YAML_PATH = Path("data/data.yaml")
WEIGHTS_SAVE_PATH = Path("models/detector/")
FINAL_WEIGHTS_NAME = "yolo_trashnet.pt"

def train_detector(epochs: int = 50, batch_size: int = 16, model_size: str = 'n'):
    """
    Trains a YOLOv8 model on the custom TrashNet dataset.
    """
    
    # 1. Load a pre-trained base model (YOLOv8 nano - 'n', or small - 's')
    # Using a pre-trained model on COCO is recommended for faster convergence.
    base_model_file = f'yolov8{model_size}.pt'
    print(f"Loading base model: {base_model_file}")
    model = YOLO(base_model_file) 
    
    # Ensure the weights directory exists
    WEIGHTS_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    # 2. Start Training
    results = model.train(
        data=str(DATA_YAML_PATH),    # Your dataset configuration file
        imgsz=640,                   # Input image size
        epochs=epochs,               # Number of training epochs (iterations over the dataset)
        batch=batch_size,            # Number of images per training step
        name='yolo_trashnet_run',    # Name for the results folder
        device='cpu',                # Use GPU (0) if available, or 'cpu'
        save=True,                   # Save the best and last model checkpoints
        exist_ok=True,               # Overwrite existing runs with the same name
        project=str(WEIGHTS_SAVE_PATH), # Save results into the models/detector folder
    )

    # The best weights are saved to a path like: models/detector/yolo_trashnet_run/weights/best.pt
    # You will manually rename and copy this file to 'yolo_trashnet.pt' later.
    
    print("\n--- Training Complete ---")
    print(f"Results saved in: {WEIGHTS_SAVE_PATH / 'yolo_trashnet_run'}")
    print("Look for the 'best.pt' file inside the 'weights' subfolder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Custom Model Trainer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--model", type=str, default='s', help="YOLOv8 model size ('n', 's', 'm', 'l', 'x'). 's' (small) is a good starting point.")
    args = parser.parse_args()
    
    train_detector(epochs=args.epochs, batch_size=args.batch, model_size=args.model)