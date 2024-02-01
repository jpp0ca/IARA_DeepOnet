"""
Utils Module

This module provides utility functions for handling device availability,
setting random seed for reproducibility, and printing information about available devices.
"""
import random

import numpy as np

import torch

def get_available_device() -> torch.device:
    """
    Get the available device for computation.

    Returns:
        torch.device: The available device, either 'cuda' (GPU) or 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_available_device():
    """ Print the available device for computation. """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU available, using CPU.")

def set_seed():
    """ Set random seed for reproducibility. """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
