import os
import shutil
import random

# ---- CONFIG ----

# Original dataset: each subfolder is a class
SOURCE_DIR = "dataset-resized"

# Output YOLO-style dataset root
OUT_DIR = "data"

# Train / Val / Test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Explicit class order -> YOLO class IDs (0..N-1)
# Make sure these exactly match your folder names
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Accepted image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# For reproducibility
random.seed(42)

# ---- SAFETY CHECK ----
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Splits must sum to 1."


def make_yolo_dirs():
    """Create data/train|val|test/images and labels directories."""
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUT_DIR, split, "images")
        lbl_dir = os.path.join(OUT_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)


def get_class_id(class_name):
    """Map class folder name to YOLO class id."""
    return CLASS_NAMES.index(class_name)


def list_images(class_dir):
    """List image files in a class directory."""
    return [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]


def split_indices(n):
    """Compute indices for train/val/test given n samples."""
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def write_yolo_label(label_path, class_id):
    """
    Write a simple YOLO label:
    one bbox covering the whole image: (x_center=0.5, y_center=0.5, w=1.0, h=1.0).
    """
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def main():
    make_yolo_dirs()

    total_counts = {"train": 0, "val": 0, "test": 0}

    # Loop over each class folder
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: class folder '{class_dir}' not found, skipping.")
            continue

        class_id = get_class_id(class_name)
        images = list_images(class_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train, n_val, n_test = split_indices(n_total)
        print(f"\nClass '{class_name}' ({class_id}): total={n_total}, "
              f"train={n_train}, val={n_val}, test={n_test}")

        # Assign images to splits
        split_files = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, file_list in split_files.items():
            for fname in file_list:
                src_img_path = os.path.join(class_dir, fname)

                # To avoid filename collisions between classes, prefix with class name
                new_fname = f"{class_name}_{fname}"

                dst_img_path = os.path.join(OUT_DIR, split, "images", new_fname)
                dst_lbl_path = os.path.join(OUT_DIR, split, "labels",
                                            os.path.splitext(new_fname)[0] + ".txt")

                # Copy image
                shutil.copy2(src_img_path, dst_img_path)
                # Create YOLO label
                write_yolo_label(dst_lbl_path, class_id)

                total_counts[split] += 1

    print("\nDone!")
    print("Total images per split:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {total_counts[split]}")


if __name__ == "__main__":
    main()
