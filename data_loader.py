"""
Data loading and preprocessing module for the SRCNN model.
Handles image loading, low-resolution generation, data augmentation, and patch creation.
"""
import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

from config import (HR_SIZE, LR_SIZE, SCALE_FACTOR, CHANNELS, 
                   PATCH_SIZE, TRAIN_HR_DIR, VAL_HR_DIR, 
                   MAX_TRAIN_IMAGES, MAX_VAL_IMAGES, TEST_HR_DIR)

def load_image(path):
    """
    Load an image from the specified path.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image in RGB format with values normalized to [0, 1]
    """
    # Read image in BGR format
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
        
    # Convert BGR to RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img

def generate_data_pairs(hr_image):
    """
    Generate low-resolution and high-resolution image pairs from a high-resolution image.
    
    Args:
        hr_image (numpy.ndarray): High-resolution input image (ground truth)
        
    Returns:
        tuple: (lr_image, hr_image) where lr_image is the generated low-resolution image
               upscaled to match hr_image dimensions
    """
    # Ensure the input image has the expected dimensions
    if hr_image.shape[:2] != HR_SIZE:
        hr_image = cv2.resize(hr_image, HR_SIZE, interpolation=cv2.INTER_CUBIC)
    
    # Generate low-resolution image by downsampling
    lr_height, lr_width = LR_SIZE
    lr_image = cv2.resize(hr_image, (lr_width, lr_height), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Upscale the low-resolution image back to original size using bicubic interpolation
    # This simulates the actual input the model will receive
    lr_image_upscaled = cv2.resize(lr_image, (HR_SIZE[1], HR_SIZE[0]), 
                                  interpolation=cv2.INTER_CUBIC)
    
    return lr_image_upscaled, hr_image

def save_image(image, path):
    """
    Save an image to the specified path.
    
    Args:
        image (numpy.ndarray): Image to save (values expected in [0, 1] range)
        path (str): Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert from [0, 1] to [0, 255] and change to BGR format
    img_to_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:  # RGB image
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_to_save)

def augment_image(image):
    """
    Apply random augmentations to the input image.
    
    Args:
        image (numpy.ndarray): Input image in RGB format
        
    Returns:
        numpy.ndarray: Augmented image
    """
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random rotation (0, 90, 180, 270 degrees)
    if random.random() > 0.5:
        k = random.randint(0, 3)
        image = np.rot90(image, k)
    
    # Random color jitter
    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        
        # Random brightness and contrast
        v = v * (0.8 + 0.4 * random.random())
        s = s * (0.8 + 0.4 * random.random())
        
        # Ensure values are in valid range
        v = np.clip(v, 0, 255)
        s = np.clip(s, 0, 255)
        
        hsv[:, :, 0] = h
        hsv[:, :, 1] = s
        hsv[:, :, 2] = v
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image

def generate_patches(image_path, patch_size, stride, scale):
    """
    Generate patches from an image for training.
    
    Args:
        image_path (str): Path to the high-resolution image
        patch_size (int): Size of the patches to extract
        stride (int): Stride for patch extraction
        scale (int): Scaling factor for super-resolution
        
    Returns:
        list: List of image patches
    """
    # Load and preprocess the image
    hr_image = load_image(image_path)
    
    # Convert to YCbCr and work on Y channel (luminance) only
    ycbcr = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
    y = ycbcr[:, :, 0]
    
    # Generate patches
    patches = []
    height, width = y.shape
    
    for y_pos in range(0, height - patch_size + 1, stride):
        for x_pos in range(0, width - patch_size + 1, stride):
            # Extract patch from Y channel
            patch = y[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]
            
            # Apply random augmentations
            if random.random() > 0.5:
                patch = augment_image(patch)
            
            # Normalize to [0, 1]
            patch = patch.astype(np.float32) / 255.0
            
            # Add channel dimension
            patch = np.expand_dims(patch, axis=-1)
            
            patches.append(patch)
    
    return patches

def load_dataset(data_dir, max_images=None):
    """
    Load a dataset of images from the specified directory.
    
    Args:
        data_dir (str): Directory containing images
        max_images (int, optional): Maximum number of images to load
        
    Returns:
        list: List of loaded images
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    image_paths = sorted(glob(os.path.join(data_dir, '*.png')) + 
                        glob(os.path.join(data_dir, '*.jpg')) +
                        glob(os.path.join(data_dir, '*.jpeg')))
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print(f"Loading {len(image_paths)} images from {data_dir}")
    images = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = load_image(path)
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return images

def prepare_training_data():
    """
    Prepare training and validation datasets.
    
    Returns:
        tuple: (train_hr_patches, val_hr_patches)
    """
    # Load training images
    train_hr_images = load_dataset(TRAIN_HR_DIR, MAX_TRAIN_IMAGES)
    val_hr_images = load_dataset(VAL_HR_DIR, MAX_VAL_IMAGES)
    
    # Generate patches for training
    print("Generating training patches...")
    train_hr_patches = []
    for img in tqdm(train_hr_images, desc="Processing training images"):
        # Convert image to YCbCr and get Y channel
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y = ycbcr[:, :, 0]
        
        # Generate patches from Y channel
        patches = generate_patches(y, PATCH_SIZE, PATCH_SIZE // 2, SCALE_FACTOR)
        train_hr_patches.extend(patches)
    
    # Generate patches for validation
    print("Generating validation patches...")
    val_hr_patches = []
    for img in tqdm(val_hr_images, desc="Processing validation images"):
        # Convert image to YCbCr and get Y channel
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y = ycbcr[:, :, 0]
        
        # Generate patches from Y channel
        patches = generate_patches(y, PATCH_SIZE, PATCH_SIZE, SCALE_FACTOR)
        val_hr_patches.extend(patches)
    
    # Convert to numpy arrays
    train_hr_patches = np.array(train_hr_patches)
    val_hr_patches = np.array(val_hr_patches)
    
    print(f"Training patches: {train_hr_patches.shape}")
    print(f"Validation patches: {val_hr_patches.shape}")
    
    return train_hr_patches, val_hr_patches
