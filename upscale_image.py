"""
Script to upscale a single image using the trained SRCNN model.
Usage: python upscale_image.py --input path/to/input.jpg --output path/to/output.png
"""
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from config import SCALE_FACTOR, MODEL_LOAD_PATH
from srcnn_model import build_srcnn

def upscale_image(model, lr_image):
    """Upscale a single low-resolution image using the SRCNN model."""
    # Convert to YCbCr and process Y channel
    if len(lr_image.shape) == 2:  # Grayscale
        y = lr_image.astype(np.float32) / 255.0
        y = np.expand_dims(y, axis=(0, -1))  # Add batch and channel dimensions
        
        # Predict using SRCNN
        sr_y = model.predict(y, verbose=0)[0, :, :, 0]
        sr_y = (sr_y * 255.0).clip(0, 255).astype(np.uint8)
        return sr_y
    
    else:  # Color image (RGB)
        ycbcr = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
        y = ycbcr[:, :, 0].astype(np.float32) / 255.0
        
        # Predict using SRCNN
        y = np.expand_dims(y, axis=(0, -1))  # Add batch and channel dimensions
        sr_y = model.predict(y, verbose=0)[0, :, :, 0]
        
        # Post-process the output
        sr_y = (sr_y * 255.0).clip(0, 255).astype(np.uint8)
        
        # Merge with original CbCr channels (upsampled if needed)
        cb = cv2.resize(ycbcr[:, :, 1], None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(ycbcr[:, :, 2], None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        
        # Merge channels and convert back to RGB
        sr_ycbcr = cv2.merge([sr_y, cb, cr])
        sr_image = cv2.cvtColor(sr_ycbcr, cv2.COLOR_YCrCb2RGB)
        
        return sr_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upscale an image using SRCNN')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save upscaled image (default: input_path_srcnn.png)')
    parser.add_argument('--model', type=str, default=MODEL_LOAD_PATH,
                       help='Path to trained model')
    parser.add_argument('--scale', type=int, default=SCALE_FACTOR,
                       help=f'Upscaling factor (default: {SCALE_FACTOR})')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_srcnn{ext}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = build_srcnn(scale_factor=args.scale)
    model.load_weights(args.model)
    
    # Load and preprocess the input image
    print(f"Loading input image from {args.input}...")
    lr_image = cv2.imread(args.input)
    if lr_image is None:
        raise ValueError(f"Could not load image from {args.input}")
    
    # Convert from BGR to RGB
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    
    # Generate low-resolution input (for demonstration)
    h, w = lr_image.shape[:2]
    lr_small = cv2.resize(lr_image, (w // args.scale, h // args.scale), 
                         interpolation=cv2.INTER_CUBIC)
    lr_image = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Upscale the image
    print("Upscaling image...")
    sr_image = upscale_image(model, lr_image)
    
    # Save the results
    print(f"Saving upscaled image to {args.output}...")
    cv2.imwrite(args.output, cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
    
    # For comparison, also save the bicubic upscaled version
    bicubic_output = args.output.replace('_srcnn.', '_bicubic.')
    bicubic = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(bicubic_output, cv2.cvtColor(bicubic, cv2.COLOR_RGB2BGR))
    
    print("\nDone!")
    print(f"- Input size: {w}x{h}")
    print(f"- Output size: {sr_image.shape[1]}x{sr_image.shape[0]}")
    print(f"- SRCNN output saved to: {args.output}")
    print(f"- Bicubic output saved to: {bicubic_output}")

if __name__ == "__main__":
    main()
