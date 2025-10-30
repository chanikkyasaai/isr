"""
Train SRCNN model for image super-resolution.
This script handles the training pipeline including data loading, model training, and saving.
"""
import os
import argparse
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import *
from data_loader import load_and_preprocess_dataset, generate_patches
from srcnn_model import build_srcnn

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def train():
    """Main training function."""
    print("Setting up directories...")
    setup_directories()
    
    print("Loading and preprocessing training data...")
    # Load and preprocess training data
    train_hr_paths = [os.path.join(TRAIN_HR_DIR, f) for f in os.listdir(TRAIN_HR_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    val_hr_paths = [os.path.join(VAL_HR_DIR, f) for f in os.listdir(VAL_HR_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Generate patches for training
    print("Generating training patches...")
    train_hr_patches = []
    for path in train_hr_paths[:MAX_TRAIN_IMAGES]:
        patches = generate_patches(path, PATCH_SIZE, PATCH_SIZE, SCALE)
        train_hr_patches.extend(patches)
    
    # Generate patches for validation
    print("Generating validation patches...")
    val_hr_patches = []
    for path in val_hr_paths[:MAX_VAL_IMAGES]:
        patches = generate_patches(path, PATCH_SIZE, PATCH_SIZE, SCALE)
        val_hr_patches.extend(patches)
    
    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_hr_patches, train_hr_patches))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_hr_patches, val_hr_patches))
    
    # Batch and prefetch for better performance
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build and compile the model
    print("Building SRCNN model...")
    model = build_srcnn(SCALE)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard callback
    log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='batch'
    )
    
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback],
        verbose=1
    )
    
    print(f"Training completed. Model saved to {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SRCNN model for image super-resolution')
    parser.add_argument('--train_dir', type=str, default=TRAIN_HR_DIR,
                      help='Directory containing training HR images')
    parser.add_argument('--val_dir', type=str, default=VAL_HR_DIR,
                      help='Directory containing validation HR images')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    TRAIN_HR_DIR = args.train_dir
    VAL_HR_DIR = args.val_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    train()
