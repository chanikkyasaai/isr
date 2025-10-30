"""
Configuration parameters for the image processing project.
"""
import os

# Image paths
TEST_IMAGE_PATH = 'data/test_hr/woman_GT.bmp'

# Noise parameters
NOISE_TYPE = 'gaussian'  # Type of noise to add
NOISE_INTENSITY = 25     # Standard deviation for Gaussian noise

# Filter parameters
WINDOW_SIZE = 5          # Size of the filter kernel (n x n)

# Output settings
OUTPUT_DIR = 'output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist
