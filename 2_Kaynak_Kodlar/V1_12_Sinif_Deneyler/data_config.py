import os

class DataConfig:
    # Source and organized paths
    SOURCE_DIR = "organized_dataset"
    PROCESSED_DIR = "processed_data"
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    
    # Split percentages
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Augmentation Params
    ROTATION_RANGE = 15
    BRIGHTNESS_RANGE = 0.2
    SHEAR_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    
    # Seed for reproducibility
    RANDOM_SEED = 42

config = DataConfig()
