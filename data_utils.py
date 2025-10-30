"""
Utility functions for image processing tasks.
"""
import cv2
import numpy as np
import os
from config import OUTPUT_DIR

def load_image(path):
    """
    Load an image from the given path, convert to RGB, and normalize to [0, 1].
    
    Args:
        path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Normalized RGB image in float32 format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
        
    # Read image in BGR format
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    
    # Convert to RGB and normalize to [0, 1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0

def add_noise(image, noise_type='gaussian', intensity=25):
    """
    Add noise to an image.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        noise_type (str): Type of noise ('gaussian')
        intensity (float): Noise intensity (standard deviation for Gaussian)
        
    Returns:
        numpy.ndarray: Noisy image (0-1 float)
    """
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = intensity / 255.0  # Scale intensity to [0,1] range
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
        noisy = image + gauss
        return np.clip(noisy, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (numpy.ndarray): Original image (0-1 float)
        processed (numpy.ndarray): Processed image (0-1 float)
        
    Returns:
        float: PSNR value in dB
    """
    # Ensure images are in [0, 1] range
    original = np.clip(original, 0.0, 1.0)
    processed = np.clip(processed, 0.0, 1.0)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original - processed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # PSNR calculation
    max_pixel = 1.0  # Since images are in [0, 1] range
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def save_image(image, filename, denormalize=True):
    """
    Save an image to the output directory.
    
    Args:
        image (numpy.ndarray): Image to save (0-1 float or 0-255 uint8)
        filename (str): Output filename
        denormalize (bool): Whether to convert from [0,1] to [0,255]
    """
    if denormalize and image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
