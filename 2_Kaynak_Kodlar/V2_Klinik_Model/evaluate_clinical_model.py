import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
import h5py
import cv2

# --- CONFIGURATION ---
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "clinical_dataset")
MODEL_PATH = "best_clinical_model_v2_1.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
CLASS_NAMES = ['Fair', 'Good', 'Poor']

# Define FocalLoss so load_model recognizes it if needed
# though we mostly use load_weights
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.math.reduce_sum(weight * cross_entropy, axis=1)

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes Grad-CAM for Functional API model.
    """
    # In Functional API, we can access layers directly from the main model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def evaluate():
    print(f"--- Loading Refined Model: {MODEL_PATH} ---")
    # Using custom_objects for FocalLoss
    with h5py.File(MODEL_PATH, 'r') as f:
        model = load_model(f, custom_objects={'FocalLoss': FocalLoss})
    
    # 1. Test Data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 2. Results
    print("\n--- Refined Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open("classification_report_v2_1.txt", "w") as f:
        f.write(report)

    # 3. Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('V2.1: Refined Clinical Model Confusion Matrix')
    plt.savefig("confusion_matrix_v2_1.png", dpi=300)

    # 4. Grad-CAM
    print("\n--- Generating Corrected Grad-CAM Heatmaps ---")
    last_conv_layer_name = "conv5_block3_out"
    selected_indices = []
    for i in range(len(CLASS_NAMES)):
        indices = np.where((y_true == i) & (y_pred == i))[0]
        if len(indices) > 0:
            selected_indices.append(indices[0])
        else:
            selected_indices.append(np.where(y_true == i)[0][0])

    plt.figure(figsize=(15, 12))
    for idx, sample_idx in enumerate(selected_indices):
        img_path = os.path.join(DATA_DIR, 'test', test_generator.filenames[sample_idx])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array_batch = np.expand_dims(img_array, axis=0)

        heatmap = get_gradcam_heatmap(img_array_batch, model, last_conv_layer_name)
        heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(np.uint8(255 * img_array), 0.6, heatmap, 0.4, 0)
        
        plt.subplot(3, 3, idx*3 + 1)
        plt.imshow(img)
        plt.title(f"Orig: {CLASS_NAMES[idx]} (Pred: {CLASS_NAMES[y_pred[sample_idx]]})")
        plt.axis('off')
        
        plt.subplot(3, 3, idx*3 + 2)
        plt.imshow(heatmap)
        plt.title("Grad-CAM")
        plt.axis('off')
        
        plt.subplot(3, 3, idx*3 + 3)
        plt.imshow(superimposed_img)
        plt.title("Overlay")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("gradcam_results_v2_1.png", dpi=300)
    print("Evaluation complete.")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        import traceback
        traceback.print_exc()
