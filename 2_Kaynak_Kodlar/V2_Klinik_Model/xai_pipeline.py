import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
import h5py

# --- CONFIGURATION ---
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "clinical_dataset")
MODEL_PATH = "best_clinical_model_v2_1.h5"
XAI_ROOT = os.path.join(BASE_DIR, "xai_results")
SAMPLES_DIR = os.path.join(XAI_ROOT, "gradcam_samples")
REPORT_FILE = os.path.join(XAI_ROOT, "xai_summary_report.md")

os.makedirs(SAMPLES_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.70
CLASS_NAMES = ['Fair', 'Good', 'Poor']

# Dummy Focal Loss for loading
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
    def call(self, y_true, y_pred):
        return tf.zeros(1)

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """
    Computes Grad-CAM for Functional API model.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def analyze_morphology(heatmap):
    """
    Heuristic to determine if focus is ICM (Center), TE (Border), or Scattered.
    """
    h, w = heatmap.shape
    center_y, center_x = h // 2, w // 2
    
    # Get coordinates of highest intensity pixels (Top 5%)
    threshold = np.percentile(heatmap, 95)
    y_coords, x_coords = np.where(heatmap >= threshold)
    
    if len(y_coords) == 0:
        return "Belirsiz", "Odak noktası tespit edilemedi."

    # Calculated Centroid
    mean_y = np.mean(y_coords)
    mean_x = np.mean(x_coords)
    std_y = np.std(y_coords)
    std_x = np.std(x_coords)
    
    # Distance from center
    dist = np.sqrt((mean_y - center_y)**2 + (mean_x - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    rel_dist = dist / max_dist
    
    # Dispersion (Scattered check)
    dispersion = (std_y + std_x) / (h + w)
    
    if dispersion > 0.15: # Arbitrary threshold for scattered focus
        return "Dağınık (Vakuol/Fragmentasyon?)", "Odak noktası dağınık, model morfolojik bütünlük kurmakta zorlanıyor veya boşluklara odaklanmış."
    elif rel_dist < 0.25:
        return "Merkezi (ICM)", "Odak noktası merkezde (ICM bölgesi), bu nedenle geleneksel morfolojik kriterlerle örtüşüyor."
    else:
        return "Çevresel (TE)", "Odak noktası dış çeperde (TE bölgesi), model trofektoderm hücrelerine odaklanıyor."

def process_and_save_xai():
    print("--- 🏁 XAI Pipeline Start ---")
    
    # 1. Load Model
    with h5py.File(MODEL_PATH, 'r') as f:
        model = load_model(f, custom_objects={'FocalLoss': FocalLoss})
    
    # 2. Setup Test Generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=True # Shuffle for random 20 samples
    )
    
    report_lines = ["# DeepEmbryo XAI Klinik Özet Raporu\n", "| Örnek ID | Tahmin | Güven | Morfolojik Analiz | Klinik Not |", "| :--- | :--- | :--- | :--- | :--- |"]
    confidence_stats = {"High": 0, "Low": 0}
    
    count = 0
    while count < 20:
        img_batch, label_batch = next(test_gen)
        img_array = img_batch[0]
        true_cls = CLASS_NAMES[np.argmax(label_batch[0])]
        
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_cls = CLASS_NAMES[pred_idx]
        conf = preds[0][pred_idx]
        
        # Grad-CAM
        heatmap = get_gradcam_heatmap(img_batch, model, "conv5_block3_out", pred_idx)
        
        # Morphological Analysis
        morph_type, morph_note = analyze_morphology(heatmap)
        
        # Visual Processing
        heatmap_resize = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resize), cv2.COLORMAP_JET)
        
        # Create Frame
        frame_color = (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 0, 255)
        if conf < CONF_THRESHOLD: confidence_stats["Low"] += 1
        else: confidence_stats["High"] += 1
        
        # Canvas Creation
        orig_px = np.uint8(255 * img_array)
        overlay = cv2.addWeighted(orig_px, 0.6, heatmap_color, 0.4, 0)
        
        # Concat images
        canvas = np.hstack([orig_px, heatmap_color, overlay])
        canvas = cv2.copyMakeBorder(canvas, 10, 80, 10, 10, cv2.BORDER_CONSTANT, value=frame_color)
        
        # Add Text
        title_text = f"Model: {pred_cls} | Guven: {conf:.2f}"
        cv2.putText(canvas, title_text, (20, canvas.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if conf < CONF_THRESHOLD:
            warning_text = "DIKKAT: Düşük güvenilirlik, manuel kontrol yapınız!"
            cv2.putText(canvas, warning_text, (20, canvas.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            ok_text = "Tahmin Guvenli Seviyededir."
            cv2.putText(canvas, ok_text, (20, canvas.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Save Image
        sample_id = f"{count:03d}"
        img_name = f"sample_{sample_id}_{pred_cls}.png"
        cv2.imwrite(os.path.join(SAMPLES_DIR, img_name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        # Report entry
        report_lines.append(f"| {sample_id} | {pred_cls} | %{conf*100:.1f} | {morph_type} | {morph_note} |")
        
        count += 1
        print(f"Processed {count}/20: {pred_cls} ({conf:.2f})")

    # 3. Finalize Markdown Report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # 4. Save Confidence Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(confidence_stats.values(), labels=confidence_stats.keys(), autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140)
    plt.title("DeepEmbryo V2.1 Prediction Confidence Distribution (Threshold: 0.70)")
    plt.savefig(os.path.join(XAI_ROOT, "confidence_distribution.png"))
    
    print(f"\n--- ✅ XAI Results Saved to: {XAI_ROOT} ---")

if __name__ == "__main__":
    try:
        process_and_save_xai()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"XAI Pipeline Error: {e}")
