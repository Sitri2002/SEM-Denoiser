import os
import random
import shutil
from glob import glob

def merge_subdirs(category_path, exts):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(category_path, split)
        if os.path.exists(split_dir):
            for f in glob(os.path.join(split_dir, '*')):
                if os.path.splitext(f)[-1].lower() in exts:
                    shutil.move(f, os.path.join(category_path, os.path.basename(f)))
            try:
                os.rmdir(split_dir)
            except OSError:
                pass

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, exts=['.jpg', '.png', '.jpeg', '.bmp']):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)

        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")

        merge_subdirs(category_path, exts)

        # Collect image paths
        image_paths = [f for f in glob(os.path.join(category_path, '*')) if os.path.splitext(f)[-1].lower() in exts]
        random.shuffle(image_paths)

        # Split indices
        n = len(image_paths)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits = {
            'train': image_paths[:n_train],
            'val': image_paths[n_train:n_train + n_val],
            'test': image_paths[n_train + n_val:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(category_path, split)
            os.makedirs(split_dir, exist_ok=True)

            for file_path in files:
                filename = os.path.basename(file_path)
                dst_path = os.path.join(split_dir, filename)
                shutil.move(file_path, dst_path)

        print(f"Split {n} images -> Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")

if __name__ == '__main__':
    random.seed(42)
    base_dataset_dir = '../dataset'  # adjust if needed
    split_dataset(base_dataset_dir)