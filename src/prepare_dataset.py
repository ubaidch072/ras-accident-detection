import os
import random
import shutil
from pathlib import Path

# ================== CONFIG ==================

# project root = parent of src folder (RAS_DETECTION)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# raw dataset location -> we only use the TRAIN folder from CCIH
# structure:
# data_raw/ccih/archive/CCIH/train/
#   ├── Acc
#   └── Nat
RAW_DIR = PROJECT_ROOT / "data_raw" / "ccih" / "archive" / "CCIH" / "train"

# where we will create our own train/val/test folders
OUT_DIR = PROJECT_ROOT / "data"

# your real folders inside RAW_DIR
# Acc = accident (crashed), Nat = natural (intact/normal)
SOURCE_CRASHED = RAW_DIR / "Acc"
SOURCE_INTACT = RAW_DIR / "Nat"

# class names we want to use in our project
# these will become folder names in data/train, data/val, data/test
CLASS_MAP = {
    "crashed": SOURCE_CRASHED,
    "intact": SOURCE_INTACT,
}

# train / val / test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# fix randomness so split is repeatable
random.seed(42)


# ================== HELPER FUNCTIONS ==================

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def collect_images(source_dir: Path):
    """Return list of all image file paths in source_dir (and subfolders)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif"}
    files = []
    for root, _, filenames in os.walk(source_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                files.append(Path(root) / fname)
    return files


# ================== MAIN SPLIT & COPY ==================

def split_and_copy():
    # create output directory structure
    for split in ["train", "val", "test"]:
        for cls in CLASS_MAP.keys():
            ensure_dir(OUT_DIR / split / cls)

    for cls_name, src_dir in CLASS_MAP.items():
        print(f"\nProcessing class '{cls_name}' from: {src_dir}")

        if not src_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {src_dir}")

        all_imgs = collect_images(src_dir)
        print(f"  Found {len(all_imgs)} images")

        if len(all_imgs) == 0:
            print("  WARNING: no images found in this folder!")
            continue

        random.shuffle(all_imgs)

        n_total = len(all_imgs)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_imgs = all_imgs[:n_train]
        val_imgs = all_imgs[n_train:n_train + n_val]
        test_imgs = all_imgs[n_train + n_val:]

        print(f"  Split -> train: {len(train_imgs)}, "
              f"val: {len(val_imgs)}, test: {len(test_imgs)}")

        def copy_list(img_list, split_name):
            for img_path in img_list:
                dest = OUT_DIR / split_name / cls_name / img_path.name
                shutil.copy2(img_path, dest)

        copy_list(train_imgs, "train")
        copy_list(val_imgs, "val")
        copy_list(test_imgs, "test")

    print("\n Done! Dataset prepared in:", OUT_DIR)


if __name__ == "__main__":
    split_and_copy()
