"""
Main script for comparing different image denoising techniques.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import (
    TEST_IMAGE_PATH, NOISE_TYPE, NOISE_INTENSITY, 
    WINDOW_SIZE, OUTPUT_DIR
)
from data_utils import load_image, add_noise, calculate_psnr, save_image
from ip_filters import (
    mean_filter, median_filter, 
    gaussian_filter, improved_adaptive_median_filter,
    bilateral_filter, non_local_means_denoising
)

def setup_plotting():
    """Set up matplotlib styles and settings."""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12

def plot_results(original, noisy, mean_filt, median_filt, gauss_filt, 
                adaptive_filt, bilateral_filt, nlm_filt, psnr_values):
    """
    Plot the comparison of original, noisy, and filtered images.
    
    Args:
        original: Original image
        noisy: Noisy image
        mean_filt: Mean filtered image
        median_filt: Median filtered image
        gauss_filt: Gaussian filtered image
        adaptive_filt: Adaptive median filtered image
        bilateral_filt: Bilateral filtered image
        nlm_filt: Non-local means filtered image
        psnr_values: Dictionary of PSNR values
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Flatten axes for easier iteration
    axes = axes.ravel()
    
    # Plot images
    images = [
        (original, 'Original Image'),
        (noisy, f'Noisy Image\n(PSNR: {psnr_values["Noisy"]:.2f} dB)'),
        (mean_filt, f'Mean Filter\n(PSNR: {psnr_values["Mean"]:.2f} dB)'),
        (median_filt, f'Median Filter\n(PSNR: {psnr_values["Median"]:.2f} dB)'),
        (gauss_filt, f'Gaussian Filter\n(PSNR: {psnr_values["Gaussian"]:.2f} dB)'),
        (adaptive_filt, f'Improved Adaptive\n(PSNR: {psnr_values["Adaptive"]:.2f} dB)'),
        (bilateral_filt, f'Bilateral Filter\n(PSNR: {psnr_values["Bilateral"]:.2f} dB)'),
        (nlm_filt, f'Non-Local Means\n(PSNR: {psnr_values["NLM"]:.2f} dB)')
    ]
    
    for idx, (img, title) in enumerate(images):
        axes[idx].imshow(np.clip(img, 0, 1))
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'comparison_results.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nResults saved to: {output_path}")

def print_psnr_table(psnr_values):
    """Print a formatted table of PSNR values."""
    print("\n" + "="*60)
    print("\t	COMPARISON OF DENOISING TECHNIQUES")
    print("-"*60)
    print(f"{'Method':<20} | {'PSNR (dB)':>15}")
    print("-"*60)
    
    for method, psnr in psnr_values.items():
        print(f"{method:<20} | {psnr:>15.2f}")
    
    print("="*60 + "\n")

def main():
    """Main function to run the image processing pipeline."""
    # Set up output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and processing images...")
    
    try:
        # Load and preprocess the image
        original = load_image(TEST_IMAGE_PATH)
        
        # Add noise to the image
        noisy = add_noise(original, NOISE_TYPE, NOISE_INTENSITY)
        
        print("\nApplying filters...")
        
        # Apply different filters
        print("  - Applying Mean Filter...")
        mean_filt = mean_filter(noisy, WINDOW_SIZE)
        
        print("  - Applying Median Filter...")
        median_filt = median_filter(noisy, WINDOW_SIZE)
        
        print("  - Applying Gaussian Filter...")
        gauss_filt = gaussian_filter(noisy, WINDOW_SIZE, sigma=1.0)
        
        print("  - Applying Improved Adaptive Median Filter...")
        adaptive_filt = improved_adaptive_median_filter(noisy, max_size=WINDOW_SIZE)
        
        print("  - Applying Bilateral Filter...")
        bilateral_filt = bilateral_filter(noisy)
        
        print("  - Applying Non-Local Means Denoising (this may take a while)...")
        nlm_filt = non_local_means_denoising(noisy)
        
        # Calculate PSNR values
        print("\nCalculating PSNR values...")
        psnr_values = {
            'Noisy': calculate_psnr(original, noisy),
            'Mean': calculate_psnr(original, mean_filt),
            'Median': calculate_psnr(original, median_filt),
            'Gaussian': calculate_psnr(original, gauss_filt),
            'Adaptive': calculate_psnr(original, adaptive_filt),
            'Bilateral': calculate_psnr(original, bilateral_filt),
            'NLM': calculate_psnr(original, nlm_filt)
        }
        
        # Print results
        print_psnr_table(psnr_values)
        
        # Save processed images
        print("Saving processed images...")
        save_image(original, 'original.png')
        save_image(noisy, 'noisy.png')
        save_image(mean_filt, 'mean_filtered.png')
        save_image(median_filt, 'median_filtered.png')
        save_image(gauss_filt, 'gaussian_filtered.png')
        save_image(adaptive_filt, 'adaptive_median_filtered.png')
        
        # Plot and save comparison
        print("Generating comparison plot...")
        setup_plotting()
        plot_results(original, noisy, mean_filt, median_filt, 
                    gauss_filt, adaptive_filt, bilateral_filt, nlm_filt, psnr_values)
        
        print("\nProcessing complete! Check the 'output' directory for results.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
