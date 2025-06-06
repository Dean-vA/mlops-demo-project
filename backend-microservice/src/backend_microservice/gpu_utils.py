import torch


def is_torch_available():
    """Check if PyTorch is available and has GPU support."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device():
    """Get the device to use for PyTorch operations."""
    if is_torch_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
