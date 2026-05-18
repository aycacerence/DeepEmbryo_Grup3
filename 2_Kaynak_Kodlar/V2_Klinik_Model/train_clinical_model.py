import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "clinical_dataset")
MODEL_PATH = os.path.join(BASE_DIR, "best_clinical_model_v2_1.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50 
NUM_CLASSES = 3

# --- 1. FOCAL LOSS ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        loss = weight * cross_entropy
        return tf.math.reduce_sum(loss, axis=1)

# --- 2. MANUAL OVERSAMPLING ---
def get_balanced_generators():
    """
    Manually balances classes by repeating minority class samples.
    """
    train_dir = os.path.join(DATA_DIR, 'train')
    classes = sorted(os.listdir(train_dir))
    
    class_paths = {cls: [] for cls in classes}
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        for f in os.listdir(cls_path):
            class_paths[cls].append(os.path.join(cls_path, f))
            
    # Find max count
    max_count = max([len(v) for v in class_paths.values()])
    print(f"\nTarget count per class: {max_count}")
    
    balanced_paths = []
    balanced_labels = []
    
    for cls, paths in class_paths.items():
        if len(paths) == 0: continue
        # Calculate how many times to repeat
        repeats = int(np.ceil(max_count / len(paths)))
        oversampled_paths = (paths * repeats)[:max_count]
        
        balanced_paths.extend(oversampled_paths)
        balanced_labels.extend([cls] * len(oversampled_paths))
        
    df_train = pd.DataFrame({
        'filename': balanced_paths,
        'class': balanced_labels
    })
    
    print("New training distribution:")
    print(df_train['class'].value_counts())

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        brightness_range=[0.9, 1.1]
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen

# --- 3. FUNCTIONAL API MODEL ---
def build_functional_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="embryo_input")
    
    # Try to load weights from parent if they exist
    weights_loc = "../resnet50_weights.h5"
    if os.path.exists(weights_loc):
        base_model = ResNet50(weights=weights_loc, include_top=False, input_tensor=inputs)
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Unfreeze Layer 4
    base_model.trainable = True
    for layer in base_model.layers[:143]:
        layer.trainable = False
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="clinical_output")(x)
    
    model = models.Model(inputs, outputs)
    return model

def train_refined():
    train_gen, val_gen = get_balanced_generators()
    model = build_functional_model()
    
    optimizer = optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-2)
    
    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        callbacks.CSVLogger("clinical_refined_training_log.csv")
    ]
    
    print("\n--- Starting Phase 2.1: Refined Clinical Training (Manual Oversampling + Focal Loss) ---")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=my_callbacks,
        verbose=1
    )
    print("\n--- Refined Training Complete! ---")

if __name__ == "__main__":
    try:
        train_refined()
    except Exception as e:
        print(f"Error: {e}")
