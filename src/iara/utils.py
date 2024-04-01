"""
Utils Module

This module provides utility functions
"""
import math
import random
import os
import datetime

import shutil
import psutil

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

def available_gpu_memory() -> float:
    """Get the gpu available memory em bytes."""

    if not torch.cuda.is_available():
        return 0

    device = torch.device("cuda")
    gpu_props = torch.cuda.get_device_properties(0)

    memory_stats = torch.cuda.memory_stats(device)
    memory_available = memory_stats["allocated_bytes.all.current"]

    memory_available = gpu_props.total_memory - memory_available
    return memory_available

def available_cpu_memory() -> float:
    """Get the cpu available memory em bytes."""
    memory = psutil.virtual_memory()
    return memory.available

def str_format_bytes(n_bytes: int) -> str:
    """ Returns string formatted for human reading """
    unity = ['B', 'KB', 'MB', 'GB', 'TB']
    cont = int(math.log(n_bytes, 1024))
    return f'{n_bytes / (1024 ** cont)} {unity[cont]}'
