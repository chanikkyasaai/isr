"""
SRCNN (Super-Resolution Convolutional Neural Network) model implementation.
This module defines the architecture and compilation of the SRCNN model.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam

from config import FILTERS_1, FILTERS_2, FILTERS_3, CHANNELS, LEARNING_RATE

def build_srcnn(scale_factor=4):
    """
    Build and compile the SRCNN model.
    
    Args:
        scale_factor (int): Scaling factor for super-resolution (not directly used in model architecture)
        
    Returns:
        tensorflow.keras.Model: Compiled SRCNN model
    """
    # Input layer - takes an upscaled low-resolution image
    input_img = Input(shape=(None, None, CHANNELS), name='input_image')
    
    # Layer 1: Patch extraction (9x9 kernel)
    x = Conv2D(filters=FILTERS_1, kernel_size=(9, 9), padding='same', 
               activation='relu', name='conv1')(input_img)
    
    # Layer 2: Non-linear mapping (1x1 kernel)
    x = Conv2D(filters=FILTERS_2, kernel_size=(1, 1), padding='same', 
               activation='relu', name='conv2')(x)
    
    # Layer 3: Reconstruction (5x5 kernel, linear activation)
    output = Conv2D(filters=FILTERS_3, kernel_size=(5, 5), padding='same', 
                   activation='linear', name='output')(x)
    
    # Create model
    model = Model(inputs=input_img, outputs=output, name='SRCNN')
    
    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    
    return model

def get_model_summary():
    """
    Utility function to print model summary.
    
    Returns:
        str: Model summary as string
    """
    model = build_srcnn()
    model.summary()
    
    # Capture the summary as string
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return '\n'.join(summary)
