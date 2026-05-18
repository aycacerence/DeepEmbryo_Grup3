import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob

def analyze_dataset(data_dir, output_dir):
    """
    Performs detailed statistical and visual analysis of the organized dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    records = []
    
    print("Analyzing images...")
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        image_files = glob.glob(os.path.join(cls_path, "*.bmp"))
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    img_array = np.array(img.convert('L')) # Convert to grayscale for pixel stats
                    
                    file_size_kb = os.path.getsize(img_path) / 1024
                    
                    records.append({
                        'class': cls,
                        'filename': os.path.basename(img_path),
                        'width': width,
                        'height': height,
                        'file_size_kb': file_size_kb,
                        'pixel_mean': np.mean(img_array),
                        'pixel_std': np.std(img_array),
                        'pixel_min': np.min(img_array),
                        'pixel_max': np.max(img_array)
                    })
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")
                
    df = pd.DataFrame(records)
    df.to_csv("dataset_statistics.csv", index=False)
    
    # 1. Class Distribution
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='class', order=sorted(classes))
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Pixel Value Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='pixel_mean', hue='class', element='step', kde=True)
    plt.title("Pixel Mean Value Distribution (Brightness)")
    plt.savefig(os.path.join(output_dir, "brightness_distribution.png"))
    plt.close()
    
    # 3. Image Dimensions Distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.5)
    plt.title("Image Dimensions Distribution")
    plt.savefig(os.path.join(output_dir, "dimensions_scatter.png"))
    plt.close()
    
    # 4. Sample Grid (6 samples per class if possible)
    # We will create a separate grid for each class's samples to avoid overcrowding
    samples_dir = os.path.join(output_dir, "class_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for cls in classes:
        cls_df = df[df['class'] == cls].sample(min(6, len(df[df['class'] == cls])))
        fig, axes = plt.subplots(1, min(6, len(cls_df)), figsize=(15, 3))
        if len(cls_df) == 1: axes = [axes]
        
        for i, (idx, row) in enumerate(cls_df.iterrows()):
            img_path = os.path.join(data_dir, cls, row['filename'])
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{row['filename']}")
            axes[i].axis('off')
        
        plt.suptitle(f"Samples for class: {cls}")
        plt.savefig(os.path.join(samples_dir, f"{cls}_samples.png"))
        plt.close()
        
    return df, classes

def generate_report(df, classes, report_path):
    """
    Generates a Markdown report based on the analysis dataframe.
    """
    total_imgs = len(df)
    class_stats = df.groupby('class').agg({
        'filename': 'count',
        'width': 'mean',
        'file_size_kb': ['mean', 'min', 'max'],
        'pixel_mean': 'mean',
        'pixel_std': 'mean'
    }).reset_index()
    
    class_stats.columns = ['Class', 'Count', 'Avg Width', 'Avg Size (KB)', 'Min Size (KB)', 'Max Size (KB)', 'Avg Brightness', 'Avg Pixel Std']
    
    class_stats['Percentage (%)'] = (class_stats['Count'] / total_imgs * 100).round(2)
    
    imbalance_ratio = class_stats['Count'].max() / class_stats['Count'].min()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Analysis Report\n\n")
        
        f.write("## 1. Overview\n")
        f.write(f"- **Total Images:** {total_imgs}\n")
        f.write(f"- **Classes:** {len(classes)}\n")
        f.write(f"- **Imbalance Ratio (Max/Min):** {imbalance_ratio:.2f}\n\n")
        
        f.write("## 2. Class Statistics\n")
        f.write(class_stats.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Data Quality Assessment\n")
        # Duplicate detection (crude based on size and brightness)
        duplicates = df[df.duplicated(subset=['width', 'height', 'file_size_kb', 'pixel_mean'], keep=False)]
        f.write(f"- **Duplicates detected:** {len(duplicates) // 2} potential sets\n")
        
        # Outliers (brightness)
        dark_imgs = df[df['pixel_mean'] < 30]
        bright_imgs = df[df['pixel_mean'] > 220]
        f.write(f"- **Very dark images (<30 mean):** {len(dark_imgs)}\n")
        f.write(f"- **Very bright images (>220 mean):** {len(bright_imgs)}\n\n")
        
        f.write("## 4. Visualizations\n")
        f.write("![Class Distribution](visualizations/class_distribution.png)\n")
        f.write("![Brightness Distribution](visualizations/brightness_distribution.png)\n")
        
        f.write("\n## 5. Recommendations for Augmentation\n")
        f.write("1. **Minority Classes:** Classes like 3CB, 4BB, 3CA are severely underrepresented. **Heavy augmentation** (Rotation, Flip, Zoom) is necessary for these.\n")
        f.write("2. **Normalization:** Brighness varies across images. **Global normalization** or Histogram Equalization might help.\n")
        f.write("3. **Resizing:** Most images have similar aspects. Resizing to 224x224 (ResNet50) will work fine.\n")

if __name__ == "__main__":
    data_dir = "organized_dataset"
    output_dir = "visualizations"
    report_path = "dataset_analysis_report.md"
    
    df, classes = analyze_dataset(data_dir, output_dir)
    generate_report(df, classes, report_path)
    print(f"Analysis complete. Report saved to {report_path}")
