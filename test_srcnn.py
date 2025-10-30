"""
Test and evaluate the trained SRCNN model on test images.
This script loads a trained model and generates super-resolved images.
"""
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from data_loader import load_image, save_image
from srcnn_model import build_srcnn

def calculate_psnr(original, compressed):
    """Calculate PSNR between original and compressed images."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def evaluate_model(model, test_dir, output_dir, max_images=None):
    """Evaluate the model on test images and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of test images
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))
    
    if max_images is not None:
        test_images = test_images[:max_images]
    
    psnr_values = []
    
    for img_path in tqdm(test_images, desc="Processing test images"):
        try:
            # Load and preprocess the HR image
            hr_img = load_image(img_path)
            
            # Generate LR image (downsample then upsample)
            lr_img = cv2.resize(hr_img, 
                              (hr_img.shape[1] // SCALE_FACTOR, hr_img.shape[0] // SCALE_FACTOR),
                              interpolation=cv2.INTER_CUBIC)
            lr_img = cv2.resize(lr_img, 
                              (hr_img.shape[1], hr_img.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
            
            # Convert to YCbCr and process Y channel
            ycbcr = cv2.cvtColor(lr_img, cv2.COLOR_RGB2YCrCb)
            y = ycbcr[:, :, 0].astype(np.float32) / 255.0
            
            # Predict using SRCNN
            y = np.expand_dims(y, axis=(0, -1))  # Add batch and channel dimensions
            sr_y = model.predict(y, verbose=0)[0, :, :, 0]
            
            # Post-process the output
            sr_y = (sr_y * 255.0).clip(0, 255).astype(np.uint8)
            
            # Merge with original CbCr channels
            sr_img = ycbcr.copy()
            sr_img[:, :, 0] = sr_y
            sr_img = cv2.cvtColor(sr_img, cv2.COLOR_YCrCb2RGB)
            
            # Calculate PSNR
            psnr = calculate_psnr(hr_img.astype(np.float32)/255.0, 
                                sr_img.astype(np.float32)/255.0)
            psnr_values.append(psnr)
            
            # Save results
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save LR, Bicubic, SR, and HR images
            save_image(lr_img, os.path.join(output_dir, f"{base_name}_lr.png"))
            save_image(cv2.resize(lr_img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, 
                               interpolation=cv2.INTER_CUBIC),
                     os.path.join(output_dir, f"{base_name}_bicubic.png"))
            save_image(sr_img, os.path.join(output_dir, f"{base_name}_srcnn.png"))
            save_image(hr_img, os.path.join(output_dir, f"{base_name}_hr.png"))
            
            # Save a comparison image
            plt.figure(figsize=(20, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
            plt.title("Low Resolution")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(
                cv2.resize(lr_img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, 
                         interpolation=cv2.INTER_CUBIC), 
                cv2.COLOR_RGB2BGR))
            plt.title("Bicubic Upscale")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
            plt.title(f"SRCNN (PSNR: {psnr:.2f} dB)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"), 
                       bbox_inches='tight', dpi=100)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Print summary
    if psnr_values:
        print(f"\nAverage PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
        print(f"Max PSNR: {np.max(psnr_values):.2f} dB")
    
    return psnr_values

def main():
    parser = argparse.ArgumentParser(description='Test SRCNN model')
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default=TEST_HR_DIR,
                       help='Directory containing test HR images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save output images')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of test images to process')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = build_srcnn(SCALE_FACTOR)
    model.load_weights(args.model_path)
    
    # Evaluate on test set
    print(f"Evaluating on images from {args.test_dir}...")
    psnr_values = evaluate_model(
        model=model,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_images=args.max_images
    )
    
    # Save PSNR values to file
    if psnr_values:
        with open(os.path.join(args.output_dir, 'psnr_results.txt'), 'w') as f:
            f.write(f"Average PSNR: {np.mean(psnr_values):.2f} dB\n")
            f.write(f"Min PSNR: {np.min(psnr_values):.2f} dB\n")
            f.write(f"Max PSNR: {np.max(psnr_values):.2f} dB\n\n")
            f.write("Per-image PSNR values:\n")
            for i, psnr in enumerate(psnr_values):
                f.write(f"Image {i+1}: {psnr:.2f} dB\n")

if __name__ == "__main__":
    main()
