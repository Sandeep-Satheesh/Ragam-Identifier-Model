import torch
import numpy as np

def convert_to_numpy_cpu(tensor):
    """Convert a PyTorch tensor to a NumPy array on CPU."""
    if tensor is None:
        return None
    return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)