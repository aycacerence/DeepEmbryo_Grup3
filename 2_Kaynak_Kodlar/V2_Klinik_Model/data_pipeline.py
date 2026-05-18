import os
import shutil
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# --- CONFIGURATION ---
SOURCE_DIR = "../EMBRIO GRADE DATASET"
BASE_TARGET_DIR = "clinical_dataset"
ALL_DATA_DIR = os.path.join(BASE_TARGET_DIR, "all_data")

# New Classes based on Gardner Quality grades
CLASS_MAPPING = {
    'Good': ['AA', 'AB', 'BA'],
    'Fair': ['BB', 'BC', 'CB'],
    'Poor': ['CC', 'CA', 'AC']
}

def identify_class(filename):
    """
    Identifies the clinical quality class based on Harf combination.
    Ignores leading numbers (expansion stage).
    """
    filename = filename.upper()
    if "CLEAVAGE" in filename:
        return None
        
    # Extract the letters (e.g., in '3AA' or '4BC')
    # Regex to find AA, AB, BA, BB, BC, CB, CC, CA, AC
    match = re.search(r'(AA|AB|BA|BB|BC|CB|CC|CA|AC)', filename)
    if match:
        found_pair = match.group(0)
        for quality, pairs in CLASS_MAPPING.items():
            if found_pair in pairs:
                return quality
    return None

def reorganize_dataset():
    """
    Copies files from source to clinical_dataset/all_data/{Good, Fair, Poor}
    """
    print("--- 1. Reorganizing Dataset to Clinical Classes ---")
    
    # Create target folders
    for cls in CLASS_MAPPING.keys():
        os.makedirs(os.path.join(ALL_DATA_DIR, cls), exist_ok=True)
        
    stats = {cls: 0 for cls in CLASS_MAPPING.keys()}
    
    # Walk through source
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith('.bmp'):
                quality_class = identify_class(file)
                if quality_class:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(ALL_DATA_DIR, quality_class, file)
                    shutil.copy2(src_path, dst_path)
                    stats[quality_class] += 1
                    
    print(f"Summary: {stats}")
    return stats

def split_dataset():
    """
    Performs Stratified Split into train/val/test folders.
    """
    print("\n--- 2. Performing Stratified Split (70/15/15) ---")
    
    all_files = []
    all_labels = []
    
    for cls in CLASS_MAPPING.keys():
        cls_dir = os.path.join(ALL_DATA_DIR, cls)
        files = [os.path.join(cls, f) for f in os.listdir(cls_dir) if f.lower().endswith('.bmp')]
        all_files.extend(files)
        all_labels.extend([cls] * len(files))
        
    # Split: Train (70%) and Temp (30%)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    
    # Split Temp: Val (15%) and Test (15%)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    splits = {
        'train': (train_files, train_labels),
        'val': (val_files, val_labels),
        'test': (test_files, test_labels)
    }
    
    # Create split folders
    for split_batch, (file_list, label_list) in splits.items():
        for filename, label in zip(file_list, label_list):
            src = os.path.join(ALL_DATA_DIR, filename)
            dst = os.path.join(BASE_TARGET_DIR, split_batch, label, os.path.basename(filename))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            
    print(f"Split completed: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return splits

def visualize_and_save_weights(splits):
    """
    Visualizes distribution and calculates class weights.
    """
    print("\n--- 3. Statistics and Class Weights ---")
    
    # Distribution for Train set
    train_labels = splits['train'][1]
    unique, counts = np.unique(train_labels, return_counts=True)
    dist_df = pd.DataFrame({'Class': unique, 'Count': counts})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=dist_df, palette='viridis')
    plt.title('V2: 3-Class Clinical Distribution (Train Set)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("new_class_distribution.png")
    print("Graph saved as new_class_distribution.png")
    
    # Calculate Class Weights
    # Get labels as integers for the sampler
    class_names = sorted(list(CLASS_MAPPING.keys()))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    y_train = [label_to_idx[l] for l in train_labels]
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    weights_dict = {int(i): float(w) for i, w in enumerate(weights)}
    with open("class_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=4)
        
    print(f"Class Weights saved: {weights_dict}")

if __name__ == "__main__":
    try:
        if os.path.exists(BASE_TARGET_DIR):
            shutil.rmtree(BASE_TARGET_DIR)
            
        stats = reorganize_dataset()
        splits = split_dataset()
        visualize_and_save_weights(splits)
        print("\n--- Success: Phase 1 Data Pipeline Complete! ---")
        
    except Exception as e:
        print(f"Critical Error during data pipeline: {e}")
