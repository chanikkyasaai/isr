# 🖼️ Advanced Image Denoising: A Comparative Study

A comprehensive analysis of image denoising techniques, featuring both traditional and advanced algorithms. This project evaluates various filters for noise reduction while preserving image details.

## 🎯 Project Overview

This project implements and compares multiple state-of-the-art image denoising techniques:

### Traditional Filters
- **Mean Filter** - Simple averaging filter
- **Median Filter** - Effective for salt-and-pepper noise
- **Gaussian Filter** - Excellent for Gaussian noise

### Advanced Filters
- **Improved Adaptive Median Filter** - Enhanced version with better edge preservation
- **Bilateral Filter** - Edge-preserving smoothing
- **Non-Local Means (NLM)** - Advanced algorithm for texture preservation

## 📊 Performance Metrics

Each filter is evaluated using:
- **PSNR (Peak Signal-to-Noise Ratio)** - Quantitative measurement of noise reduction
- **Visual Quality** - Subjective assessment of edge and detail preservation

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

1. **Prepare your image**:
   - Place your high-resolution test image in the `data/test_hr/` directory
   - Ensure the image is in a common format (JPEG, PNG, BMP)

2. **Configuration** (optional):
   Edit `config.py` to adjust parameters:
   ```python
   TEST_IMAGE_PATH = 'data/test_hr/your_image.jpg'  # Path to your image
   NOISE_TYPE = 'gaussian'  # Type of noise to add
   NOISE_INTENSITY = 25     # Noise intensity (standard deviation)
   WINDOW_SIZE = 5          # Base window size for filters (odd number)
   ```

3. **Run the analysis**:
   ```bash
   python main_analysis.py
   ```

## 📂 Output

The script generates the following in the `output/` directory:
- `original.png` - Original input image
- `noisy.png` - Noisy version of the image
- `*_filtered.png` - Processed images from each filter
- `comparison_results.png` - Side-by-side comparison of all results

## 📈 Results Interpretation

- **PSNR Values**: Higher values indicate better noise reduction
- **Visual Quality**: Check the `comparison_results.png` to evaluate:
  - Edge preservation
  - Detail retention
  - Artifact introduction

## 🏆 Performance Comparison

Typical PSNR results (higher is better):

| Filter Type          | PSNR (dB) | Best For                     |
|----------------------|-----------|------------------------------|
| Noisy Image          | ~20-22    | -                            |
| Mean Filter          | ~28-30    | Quick processing             |
| Median Filter        | ~28-30    | Salt-and-pepper noise        |
| Gaussian Filter      | ~29-31    | Gaussian noise               |
| Adaptive Median      | ~29-31    | Mixed noise types            |
| Bilateral Filter     | ~28-30    | Edge preservation            |
| Non-Local Means      | ~25-27    | Texture preservation         |

## 🛠️ Customization

### Adding New Filters
1. Add your filter function to `ip_filters.py`
2. Update `main_analysis.py` to include your filter in the processing pipeline

### Adjusting Parameters
Modify the filter parameters in `main_analysis.py` for different results:
- For Bilateral Filter: Adjust `d`, `sigma_color`, `sigma_space`
- For NLM: Adjust `h`, `template_window_size`, `search_window_size`

## 📚 Dependencies

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- tqdm (for progress bars)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or suggestions, please open an issue or contact the project maintainers.
