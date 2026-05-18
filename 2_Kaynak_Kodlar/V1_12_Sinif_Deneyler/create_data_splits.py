import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split
from data_config import config

def create_splits(source_dir, target_dir):
    """
    Splits images into train/val/test folders using a stratified approach.
    """
    # 1. Gather all files and labels
    all_data = []
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        for img in images:
            if img.lower().endswith('.bmp'):
                all_data.append({'filename': img, 'class': cls, 'path': os.path.join(cls_path, img)})

    filenames = [d['path'] for d in all_data]
    labels = [d['class'] for d in all_data]

    # 2. Split with Stratification
    # First split into Train and Temp (Val + Test)
    # Since some classes have very few samples (e.g., 3), train_test_split might complain.
    # We will manually ensure at least 1 sample per split if total >= 3.
    
    train_files, val_files, test_files = [], [], []
    
    for cls in classes:
        cls_files = [d['path'] for d in all_data if d['class'] == cls]
        random.seed(config.RANDOM_SEED)
        random.shuffle(cls_files)
        
        n = len(cls_files)
        if n >= 3:
            # Traditional split roughly matching 70/15/15
            n_val = max(1, int(n * config.VAL_SPLIT))
            n_test = max(1, int(n * config.TEST_SPLIT))
            n_train = n - n_val - n_test
            
            train_files.extend(cls_files[:n_train])
            val_files.extend(cls_files[n_train:n_train+n_val])
            test_files.extend(cls_files[n_train+n_val:])
        elif n == 2:
            train_files.append(cls_files[0])
            val_files.append(cls_files[1]) # Put one in val
        else: # n == 1
            train_files.append(cls_files[0])

    print(f"Split results: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 3. Create Folder Structure and Copy
    split_map = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    split_info = {}

    for split_name, files in split_map.items():
        split_path = os.path.join(target_dir, split_name)
        os.makedirs(split_path)
        
        for f in files:
            cls = os.path.basename(os.path.dirname(f))
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)
            shutil.copy2(f, os.path.join(split_path, cls, os.path.basename(f)))
            
            split_info[os.path.basename(f)] = split_name

    # 4. Save splits info
    with open("data_splits.json", "w") as f:
        json.dump(split_info, f, indent=4)
        
    return split_map

if __name__ == "__main__":
    create_splits(config.SOURCE_DIR, config.PROCESSED_DIR)
    print("Data splits created successfully.")
