import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from data_config import config

def create_model(num_classes):
    """
    Creates a ResNet50 model with a custom classification head.
    """
    # 1. Base Model: ResNet50
    # Loading weights from local file downloaded via curl
    base_model = ResNet50(
        weights='resnet50_weights.h5',
        include_top=False,
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # 2. Custom Classification Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def count_parameters(model):
    """
    Displays the number of trainable and non-trainable parameters.
    """
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

if __name__ == "__main__":
    num_classes = 13 # Including Cleavage as found in dataset analysis
    model = create_model(num_classes)
    model.summary()
    count_parameters(model)
    
    # Save model summary to file
    with open("model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
