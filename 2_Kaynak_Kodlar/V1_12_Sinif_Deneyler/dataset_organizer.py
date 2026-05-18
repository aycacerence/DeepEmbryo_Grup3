import os
import shutil
import re
from PIL import Image

def organize_dataset(source_dir, target_dir):
    """
    Organizes the embryo dataset based on filenames and identifies corrupted images.
    """
    classes = [
        '3AA', '3AB', '3BA', '3BB', '3BC', '3CA', '3CB', '3CC', 
        '4AA', '4AB', '4BA', '4BB', 'Cleavage'
    ]
    
    # Create target directories
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for cls in classes:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)
    
    stats = {cls: 0 for cls in classes}
    corrupted_files = []
    total_files = 0
    
    # Regex for matching the class in filename (e.g., 3AA (10).bmp -> 3AA)
    # Also handle variants like "Klivaj Grade I.bmp"
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.lower().endswith('.bmp'):
                continue
            
            total_files += 1
            source_path = os.path.join(root, filename)
            
            # Determine class
            file_class = None
            
            # Check for Cleavage first
            if 'klivaj' in filename.lower() or 'cleavage' in filename.lower() or 'cleavage' in root.lower():
                file_class = 'Cleavage'
            else:
                # Try to find class label in filename
                match = re.search(r'(3AA|3AB|3BA|3BB|3BC|3CA|3CB|3CC|4AA|4AB|4BA|4BB)', filename, re.IGNORECASE)
                if match:
                    file_class = match.group(0).upper()
            
            if file_class:
                # Validate image
                try:
                    with Image.open(source_path) as img:
                        img.verify() # Verify file integrity
                    
                    target_path = os.path.join(target_dir, file_class, filename)
                    shutil.copy2(source_path, target_path)
                    stats[file_class] += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    corrupted_files.append(source_path)
            else:
                print(f"Unidentified class for file: {filename}")
    
    return stats, corrupted_files, total_files

if __name__ == "__main__":
    source_path = "EMBRIO GRADE DATASET"
    target_path = "organized_dataset"
    
    print(f"Starting organization from '{source_path}' to '{target_path}'...")
    
    stats, corrupted, total = organize_dataset(source_path, target_path)
    
    print("\n--- Organization Report ---")
    print(f"Total files found: {total}")
    print(f"Successfully organized: {sum(stats.values())}")
    print(f"Corrupted files: {len(corrupted)}")
    
    print("\nClass Distribution:")
    for cls, count in stats.items():
        print(f"  {cls}: {count}")
    
    if corrupted:
        print("\nCorrupted Files List:")
        for f in corrupted:
            print(f"  {f}")
    
    # Check for imbalance
    max_cls = max(stats, key=stats.get)
    min_cls = min([c for c in stats if stats[c] > 0], key=lambda k: stats[k])
    
    if stats[max_cls] > 0 and stats[min_cls] > 0:
        ratio = stats[max_cls] / stats[min_cls]
        if ratio > 2.0:
            print(f"\n[WARNING] Critical Class Imbalance Detected!")
            print(f"Ratio {max_cls}/{min_cls} = {ratio:.2f}")
