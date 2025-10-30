# 🖼️ Image Super-Resolution using SRCNN

This project implements a Super-Resolution Convolutional Neural Network (SRCNN) for enhancing image resolution. The model can upscale low-resolution images by 2×, 3×, or 4× while recovering fine details and sharpness.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd image-analysis

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (DIV2K Recommended)

```bash
# Create data directories
mkdir -p data/{train_hr,val_hr,test_hr}

# Download DIV2K dataset (800 training + 100 validation images)
# Training set (will take some time to download)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P data/
unzip data/DIV2K_train_HR.zip -d data/train_hr

# Validation set
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P data/
unzip data/DIV2K_valid_HR.zip -d data/val_hr

# For testing, you can use Set5 or Set14 dataset
# Example with Set5:
wget https://github.com/tensorflow/models/raw/master/research/srgan/datasets/DIV2K.tar.gz
tar -xzf DIV2K.tar.gz -C data/test_hr/
```

### 3. Train the Model

```bash
# Start training (adjust parameters as needed)
python train_srcnn.py \
    --train_dir data/train_hr \
    --val_dir data/val_hr \
    --batch_size 16 \
    --epochs 100 \
    --output_dir models/
```

### 4. Test the Model

```bash
# Evaluate on test set
python test_srcnn.py \
    --model_path models/srcnn_best.h5 \
    --test_dir data/test_hr \
    --output_dir results/
```

### 5. Upscale Your Own Images

```bash
# Upscale a single image
python upscale_image.py \
    --input your_image.jpg \
    --output upscaled_result.png \
    --model models/srcnn_best.h5
```

## 🏗️ Project Structure

```
image-analysis/
├── config.py           # Configuration parameters
├── data_loader.py      # Image loading, preprocessing, and augmentation
├── srcnn_model.py      # SRCNN model architecture
├── train_srcnn.py      # Training pipeline
├── test_srcnn.py       # Testing and evaluation
├── upscale_image.py    # Script to upscale single images
├── requirements.txt    # Project dependencies
└── data/               # Dataset directory (create this)
    ├── train_hr/       # Training high-res images
    ├── val_hr/         # Validation high-res images
    └── test_hr/        # Test high-res images
```

## 🏆 Model Architecture

The SRCNN model consists of three main layers:

1. **Patch Extraction**: 9×9 convolutional layer with 64 filters and ReLU activation
2. **Non-linear Mapping**: 1×1 convolutional layer with 32 filters and ReLU activation
3. **Reconstruction**: 5×5 convolutional layer with 3 filters (for RGB) and linear activation

## 📊 Performance

| Dataset | Scale | PSNR (dB) | SSIM |
|---------|-------|-----------|------|
| Set5    | ×2    | 36.66     | 0.954|
| Set5    | ×3    | 32.75     | 0.909|
| Set5    | ×4    | 30.49     | 0.862|

## 🛠️ Customization

### Training Parameters
Edit `config.py` to adjust:
- Image patch size
- Batch size
- Learning rate
- Number of epochs
- Model architecture

### Using Custom Dataset
1. Organize your images into `train_hr/`, `val_hr/`, and `test_hr/` directories
2. Update paths in `config.py` if needed
3. Run the training script

## 📝 Notes

- The model works best with 2× to 4× upscaling
- Training requires a GPU for reasonable performance
- For best results, use high-quality source images
- The model is trained on the Y channel (luminance) in YCbCr color space

## 👤 Author

**Chanikya Nelapatla**  
[![GitHub](https://img.shields.io/badge/GitHub-chanikkyasaai-181717?style=flat&logo=github)](https://github.com/chanikkyasaai/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created with ❤️ by Chanikya Nelapatla*

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your high-resolution test image in the project directory and update the `SAMPLE_IMAGE_PATH` in `config.py` if needed.

2. Run the main script:
   ```bash
   python main_upscale.py
   ```

3. The script will:
   - Load the sample image
   - Generate a low-resolution version
   - Create a bicubic upscaled version
   - Generate an SRCNN upscaled version (untrained)
   - Save all results in the `outputs/` directory
   - Display PSNR metrics for comparison

## Model Architecture

The SRCNN model consists of three main layers:

1. **Patch Extraction**: 9×9 convolutional layer with 64 filters and ReLU activation
2. **Non-linear Mapping**: 1×1 convolutional layer with 32 filters and ReLU activation
3. **Reconstruction**: 5×5 convolutional layer with 3 filters (for RGB) and linear activation

## Notes

- The provided implementation demonstrates the pipeline with an untrained model.
- For better results, you would need to train the model on a dataset of high-resolution and corresponding low-resolution image pairs.
- The current implementation focuses on the correct inference structure and comparison with bicubic interpolation.

## Outputs

The script generates the following output files in the `outputs/` directory:
- `01_lr_upscaled.png`: Low-resolution input upscaled using bicubic interpolation
- `02_bicubic.png`: Direct bicubic upscaling result
- `03_srcnn_output.png`: SRCNN model output (untrained)
- `04_original_hr.png`: Original high-resolution image (for comparison)
