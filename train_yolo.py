import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Keep this for environment compatibility

from ultralytics import YOLO
import argparse
from pathlib import Path
import yaml # Import for creating the data.yaml file

# Define paths relative to the script execution location
BASE_DIR = Path('.')
DATA_YAML_PATH = BASE_DIR / "data" / "data.yaml"
WEIGHTS_SAVE_PATH = BASE_DIR / "models" / "detector"

# --- Dataset and Class Configuration ---
# Image and Annotation locations (relative to script)
IMAGE_ROOT = BASE_DIR / 'data' / 'cleaned_data'
YOLO_ANNOTATION_ROOT = BASE_DIR / 'data' / 'yolo_annotations' # New folder from conversion

# This must match the list in your 'convert_xml_to_yolo.py' script EXACTLY
CLASS_NAMES = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash' 
    # Ensure this is correct and complete
]

def create_data_yaml(yaml_path: Path):
    """Creates the data.yaml file required by YOLOv8."""
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_config = {
        # Path to the dataset root folder (where 'train' and 'test' folders are)
        # We point to the parent of 'cleaned_data' to make paths relative
        'path': str(IMAGE_ROOT.resolve()), 
        
        # Relative paths to the train/validation images
        # NOTE: YOLOv8 automatically looks for annotations in the 'labels' folder 
        # parallel to the 'images' folder. 
        # Our images are in 'cleaned_data/train', so we must configure YOLO to 
        # look for annotations in 'yolo_annotations/train'.
        
        # To handle this, we set the 'path' to the root where the image folders are, 
        # and point 'train' and 'val' to the image folder names.
        
        'train': 'train', 
        'val': 'test', # Using test data as validation is common practice for a quick start
        
        # Number of classes
        'nc': len(CLASS_NAMES), 
        
        # Class names (must be ordered to match the class index in annotations)
        'names': CLASS_NAMES
    }

    # IMPORTANT WORKAROUND: YOLOv8 expects images in /images and annotations in /labels.
    # To use our custom 'cleaned_data' and 'yolo_annotations' folders, we must 
    # pass the full path of the image folders and create symlinks or copy the annotation files.
    # A simpler approach is to structure the data exactly as YOLO expects:
    # `dataset/images/train`, `dataset/labels/train`, etc.
    
    # A cleaner approach for your existing structure is to make the paths explicit:
    
    data_config_explicit = {
        'path': str(BASE_DIR.resolve()), # Set base path to current directory
        'train': str(IMAGE_ROOT / 'train'), # Full path to train images
        'val': str(IMAGE_ROOT / 'test'),    # Full path to test images
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
        # This is the non-standard part: YOLOv8 does not allow specifying label path 
        # directly in the YAML for *training*. It is hardcoded to look for 'labels' 
        # parallel to 'images'.
        # The easiest fix is to make your structure compliant OR use the 'dataset/images/labels'
        # structure and pass the root path.
        
        # **The most reliable method for your structure is to copy/symlink annotations:**
        # For this script to work, you must copy the .txt files from 
        # 'yolo_annotations/train' into 'cleaned_data/train' but rename the folder to 'labels'.
        # That is, copy 'yolo_annotations/train/*.txt' to 'cleaned_data/train/labels/*.txt'
        # and 'yolo_annotations/test/*.txt' to 'cleaned_data/test/labels/*.txt'
        
        # Since I can't run file operations, I'll structure the YAML for the **compliant structure**:
        # ASSUMING you have a structure like:
        # DATASET_ROOT/
        # ├── images/
        # │   ├── train/  (all train images)
        # │   └── val/    (all test/validation images)
        # └── labels/
        #     ├── train/  (all train .txt annotations)
        #     └── val/    (all test/validation .txt annotations)
        
        # We will keep the paths relative to the current directory for simplicity.
        # IF you use the structure above, then:
        
        'path': str((BASE_DIR / 'dataset_root').resolve()), # e.g., if you created a new folder 'dataset_root'
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
        
    }
    
    # For now, let's use the simplest configuration assuming you will restructure or symlink.
    # We will point 'train' and 'val' directly to the image folders.
    # For this to work, you *must* ensure that a subfolder named 'labels'
    # exists in 'cleaned_data/train' and 'cleaned_data/test', containing your YOLO .txt files.

    yaml_content = {
        'path': str(IMAGE_ROOT.resolve()),
        'train': 'train',
        'val': 'test', 
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Created data.yaml at: {yaml_path}")
    print("🚨 Ensure you have a 'labels' subfolder inside 'cleaned_data/train' and 'cleaned_data/test' containing the YOLO .txt files!")


def train_detector(epochs: int = 50, batch_size: int = 16, model_size: str = 'n'):
    """
    Trains a YOLOv8 model on the custom dataset.
    """
    
    # 0. Generate data.yaml before starting training
    create_data_yaml(DATA_YAML_PATH)
    
    # 1. Load a pre-trained base model
    base_model_file = f'yolov8{model_size}.pt'
    print(f"\nLoading base model: {base_model_file}")
    model = YOLO(base_model_file) 
    
    # Ensure the weights directory exists
    WEIGHTS_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    # 2. Start Training
    results = model.train(
        data=str(DATA_YAML_PATH), # Your dataset configuration file
        imgsz=640, # Input image size
        epochs=epochs, # Number of training epochs
        batch=batch_size, # Number of images per training step
        name='yolo_trashnet_run', # Name for the results folder
        device=0, # Use GPU (0) if available, or 'cpu'
        save=True, # Save the best and last model checkpoints
        exist_ok=True,  # Overwrite existing runs with the same name
        project=str(WEIGHTS_SAVE_PATH), # Save results into the models/detector folder
        # The `data` argument now points to the dynamically generated data.yaml
    )
    
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