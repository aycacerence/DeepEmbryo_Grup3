import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from data_config import config

def get_data_generators():
    """
    Creates training, validation and test generators for TensorFlow/Keras.
    """
    # 1. Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        brightness_range=[1.0 - config.BRIGHTNESS_RANGE, 1.0 + config.BRIGHTNESS_RANGE],
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        vertical_flip=config.VERTICAL_FLIP,
        fill_mode='nearest'
    )

    # 2. Rescaling for val/test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # 3. Flow from directories
    train_generator = train_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DIR, 'train'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DIR, 'validation'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DIR, 'test'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def calculate_class_weights(train_dir):
    """
    Calculates weights to handle imbalance.
    """
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    labels = []
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(train_dir, cls)
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith('.bmp')])
        labels.extend([i] * count)
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return {i: float(weight) for i, weight in enumerate(weights)}

if __name__ == "__main__":
    print("Testing TensorFlow Preprocessing Pipeline...")
    train_dir = os.path.join(config.PROCESSED_DIR, 'train')
    
    if os.path.exists(train_dir):
        weights = calculate_class_weights(train_dir)
        print(f"Calculated Class Weights: {weights}")
        
        with open("class_weights.json", "w") as f:
            json.dump(weights, f, indent=4)
            
        train_gen, val_gen, test_gen = get_data_generators()
        print(f"Train samples: {train_gen.samples}")
        print(f"Classes: {train_gen.class_indices}")
    else:
        print("Please run create_data_splits.py first.")
