"""
Image processing filters for noise reduction.
"""
import numpy as np
import cv2
from tqdm import tqdm
from scipy import ndimage

def mean_filter(image, size):
    """
    Apply a mean filter to the image.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        size (int): Size of the square kernel
        
    Returns:
        numpy.ndarray: Filtered image (0-1 float)
    """
    # Convert to 8-bit for OpenCV
    img_8u = (image * 255).astype(np.uint8)
    
    # Apply mean filter
    filtered = cv2.blur(img_8u, (size, size))
    
    # Convert back to float and normalize
    return filtered.astype(np.float32) / 255.0

def median_filter(image, size):
    """
    Apply a median filter to the image.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        size (int): Size of the square kernel (must be odd)
        
    Returns:
        numpy.ndarray: Filtered image (0-1 float)
    """
    # Convert to 8-bit for OpenCV
    img_8u = (image * 255).astype(np.uint8)
    
    # Apply median filter
    filtered = cv2.medianBlur(img_8u, size)
    
    # Convert back to float and normalize
    return filtered.astype(np.float32) / 255.0

def gaussian_filter(image, size, sigma=0):
    """
    Apply a Gaussian filter to the image.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        size (int): Size of the square kernel (must be odd)
        sigma (float): Standard deviation in X and Y directions
        
    Returns:
        numpy.ndarray: Filtered image (0-1 float)
    """
    # Convert to 8-bit for OpenCV
    img_8u = (image * 255).astype(np.uint8)
    
    # Apply Gaussian filter
    filtered = cv2.GaussianBlur(img_8u, (size, size), sigma)
    
    # Convert back to float and normalize
    return filtered.astype(np.float32) / 255.0

def improved_adaptive_median_filter(image, max_size=7):
    """
    Enhanced Adaptive Median Filter with better edge preservation.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        max_size (int): Maximum window size (must be odd and >= 3)
        
    Returns:
        numpy.ndarray: Filtered image (0-1 float)
    """
    # Convert to 8-bit for processing
    img_8u = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Process each channel separately
    channels = []
    for c in range(img_8u.shape[2]):
        channel = img_8u[:, :, c]
        
        # First pass: standard median filter to remove salt-and-pepper noise
        filtered = cv2.medianBlur(channel, 3)
        
        # Second pass: apply adaptive median with edge preservation
        result = np.zeros_like(filtered)
        rows, cols = filtered.shape
        
        for i in range(rows):
            for j in range(cols):
                window_size = 3
                pixel = filtered[i, j]
                
                while window_size <= max_size:
                    half = window_size // 2
                    
                    # Get window with boundary checks
                    x_min, x_max = max(0, i-half), min(rows, i+half+1)
                    y_min, y_max = max(0, j-half), min(cols, j+half+1)
                    window = filtered[x_min:x_max, y_min:y_max]
                    
                    # Calculate statistics
                    z_min, z_med, z_max = np.min(window), np.median(window), np.max(window)
                    
                    # Level A: Is median within min-max range?
                    if z_min < z_med < z_max:
                        # Level B: Is current pixel within min-max range?
                        if z_min < pixel < z_max:
                            result[i, j] = pixel  # Not noise, keep original
                        else:
                            result[i, j] = z_med  # Replace with median
                        break
                    else:
                        window_size += 2  # Increase window size
                        
                        if window_size > max_size:
                            result[i, j] = z_med  # Max size reached, use median
                            break
        
        # Final pass: apply mild Gaussian blur to smooth while preserving edges
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        channels.append(result)
    
    # Combine channels and normalize
    filtered = np.stack(channels, axis=2)
    return filtered.astype(np.float32) / 255.0

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        d (int): Diameter of pixel neighborhood
        sigma_color (float): Filter sigma in color space
        sigma_space (float): Filter sigma in coordinate space
        
    Returns:
        numpy.ndarray: Filtered image (0-1 float)
    """
    # Convert to 8-bit for OpenCV
    img_8u = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(img_8u, d, sigma_color, sigma_space)
    
    # Convert back to float and normalize
    return filtered.astype(np.float32) / 255.0

def non_local_means_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-Local Means Denoising for better texture preservation.
    
    Args:
        image (numpy.ndarray): Input image (0-1 float)
        h (float): Parameter regulating filter strength
        template_window_size (int): Size of the template patch
        search_window_size (int): Size of the window to search for similar patches
        
    Returns:
        numpy.ndarray: Denoised image (0-1 float)
    """
    # Convert to 8-bit for OpenCV
    img_8u = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_8u,
        None,
        h,
        h,
        template_window_size,
        search_window_size
    )
    
    # Convert back to float and normalize
    return denoised.astype(np.float32) / 255.0
    
    # Convert back to float and normalize
    return output.astype(np.float32) / 255.0
