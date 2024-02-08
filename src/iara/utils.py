"""
Utils Module

This module provides utility functions
"""
import random
import os
import shutil
import datetime

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

def backup_folder(base_dir, time_str_format = "%Y%m%d-%H%M%S"):
    """Method to backup all files in a folder in a timestamp based folder

    Args:
        base_dir (_type_): Directory to backup
        time_str_format (str, optional): Time string format for the folder.
            Defaults to "%Y%m%d-%H%M%S".
    """
    backup_dir = os.path.join(base_dir, datetime.datetime.now().strftime(time_str_format))
    os.makedirs(backup_dir)

    contents = os.listdir(base_dir)
    for item in contents:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            try:
                datetime.datetime.strptime(item, time_str_format)
                continue
            except ValueError:
                pass
        shutil.move(item_path, backup_dir)
