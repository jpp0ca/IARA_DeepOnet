import typing
import random

import numpy as np

import torch

def get_available_device() -> typing.Union[str, torch.device]:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_available_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU available, using CPU.")

def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
