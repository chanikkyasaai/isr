"""
Main script for the SRCNN-based image super-resolution project.
This script demonstrates the complete pipeline from loading an image to comparing
bicubic upscaling with the SRCNN model's output.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from config import SCALE_FACTOR, HR_SIZE, SAMPLE_IMAGE_PATH, OUTPUT_DIR
from data_loader import load_image, generate_data_pairs, save_image
from srcnn_model import build_srcnn, get_model_summary

def calculate_psnr(original, compressed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (numpy.ndarray): Original image
        compressed (numpy.ndarray): Processed/compressed image
        
    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal
        return 100
    max_pixel = 1.0  # Since we're working with normalized images [0,1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("SRCNN Image Super-Resolution Demo")
    print("=" * 50)
    
    # Load and preprocess the high-resolution image
    try:
        print(f"Loading sample image from: {SAMPLE_IMAGE_PATH}")
        hr_image = load_image(SAMPLE_IMAGE_PATH)
        print(f"Original image size: {hr_image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please ensure the sample image exists at the specified path.")
        return
    
    # Generate low-resolution and high-resolution pairs
    print("\nGenerating low-resolution image...")
    lr_image_upscaled, hr_image = generate_data_pairs(hr_image)
    
    # Create bicubic upscaled version for comparison
    print("Creating bicubic upscaled version...")
    bicubic_upscaled = cv2.resize(
        cv2.resize(hr_image, (HR_SIZE[1]//SCALE_FACTOR, HR_SIZE[0]//SCALE_FACTOR), 
                  interpolation=cv2.INTER_CUBIC),
        (HR_SIZE[1], HR_SIZE[0]), 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Build and load the SRCNN model
    print("\nBuilding SRCNN model...")
    model = build_srcnn(SCALE_FACTOR)
    print("\nModel Summary:")
    print("-" * 50)
    print(get_model_summary())
    
    # Prepare input for prediction (add batch dimension)
    input_img = np.expand_dims(lr_image_upscaled, axis=0)
    
    # Generate SRCNN output (using untrained model for demonstration)
    print("\nGenerating SRCNN output (using untrained model)...")
    srcnn_output = model.predict(input_img)[0]
    
    # Calculate PSNR values
    bicubic_psnr = calculate_psnr(hr_image, bicubic_upscaled)
    srcnn_psnr = calculate_psnr(hr_image, srcnn_output)
    
    print("\nResults:")
    print("-" * 50)
    print(f"Bicubic PSNR: {bicubic_psnr:.2f} dB")
    print(f"SRCNN PSNR: {srcnn_psnr:.2f} dB")
    
    # Save results
    print("\nSaving results...")
    save_image(lr_image_upscaled, os.path.join(OUTPUT_DIR, '01_lr_upscaled.png'))
    save_image(bicubic_upscaled, os.path.join(OUTPUT_DIR, '02_bicubic.png'))
    save_image(srcnn_output, os.path.join(OUTPUT_DIR, '03_srcnn_output.png'))
    save_image(hr_image, os.path.join(OUTPUT_DIR, '04_original_hr.png'))
    
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nNote: The SRCNN model is untrained. For better results, train the model with a dataset.")

if __name__ == "__main__":
    main()
