"""
Configuration parameters for the SRCNN (Super-Resolution Convolutional Neural Network) model.
"""

# Image configuration
SCALE_FACTOR = 4  # 4x upscaling factor
HR_SIZE = (96, 96)  # High-resolution image size (height, width)
LR_SIZE = (24, 24)  # Low-resolution image size (1/4 of HR size)
CHANNELS = 3  # Number of color channels (RGB)

# Model architecture parameters
FILTERS_1 = 64  # Number of filters in the first convolutional layer
FILTERS_2 = 32  # Number of filters in the second convolutional layer
FILTERS_3 = 3   # Number of filters in the output layer (3 for RGB)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 0.001
STEPS_PER_EPOCH = 100  # Number of batches per epoch
VALIDATION_SPLIT = 0.2  # 20% of data for validation
MAX_TRAIN_IMAGES = 800  # Maximum number of training images to use
MAX_VAL_IMAGES = 100    # Maximum number of validation images to use

# Dataset parameters
TRAIN_HR_DIR = 'data/train_hr'  # Directory containing training HR images
VAL_HR_DIR = 'data/val_hr'      # Directory containing validation HR images
TEST_HR_DIR = 'data/test_hr'    # Directory containing test HR images
PATCH_SIZE = 96  # Size of image patches for training
SCALE = 4        # Scaling factor

# Model saving
MODEL_SAVE_PATH = 'models/srcnn_model.h5'  # Path to save trained model
MODEL_LOAD_PATH = 'models/srcnn_model.h5'  # Path to load trained model

# Paths
SAMPLE_IMAGE_PATH = 'sample_hr.jpg'  # Path to sample high-resolution image
OUTPUT_DIR = 'outputs'  # Directory to save output images
