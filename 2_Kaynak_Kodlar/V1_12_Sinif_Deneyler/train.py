import os
import json
import tensorflow as tf
from data_preprocessing import get_data_generators
from model_architecture import create_model
from data_config import config

def train_model(epochs=100, batch_size=16):
    """
    Orchestrates the two-stage training loop.
    """
    # 1. Load Data
    train_gen, val_gen, test_gen = get_data_generators()
    num_classes = train_gen.num_classes
    
    # Load Class Weights
    with open("class_weights.json", "r") as f:
        class_weights = json.load(f)
    # Convert keys to int for Keras
    class_weights = {int(k): v for k, v in class_weights.items()}

    # 2. Build Model
    model = create_model(num_classes)
    
    # 3. Step 1: Warm-up (Base frozen)
    print("\n--- Starting Stage 1: Warm-up ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/best_model_warmup.h5', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger('logs/warmup_log.csv')
    ]
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs/warmup/train', exist_ok=True)
    os.makedirs('logs/warmup/validation', exist_ok=True)
    os.makedirs('logs/finetuning/train', exist_ok=True)
    os.makedirs('logs/finetuning/validation', exist_ok=True)
    
    history_warmup = model.fit(
        train_gen,
        epochs=10, # Short warm-up as requested
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 4. Step 2: Fine-Tuning (Unfreeze base)
    print("\n--- Starting Stage 2: Fine-tuning ---")
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Optional: Unfreeze only from a certain layer (e.g., top-50)
    # fine_tune_at = 140
    # for layer in model.layers[0].layers[:fine_tune_at]:
    #     layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/best_model_final.h5', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8),
        tf.keras.callbacks.CSVLogger('logs/finetuning_log.csv')
    ]
    
    history_ft = model.fit(
        train_gen,
        epochs=epochs,
        initial_epoch=history_warmup.epoch[-1] + 1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_ft
    )
    
    return history_ft

if __name__ == "__main__":
    train_model(epochs=100)
