import logging
from typing import Any, Dict

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_torch_available():
    """Check if PyTorch is available and has GPU support."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False

def get_device():
    """Get the appropriate device (GPU or CPU) for model execution.

    Returns:
        str: Device string ('cuda' or 'cpu')
    """

    if torch.cuda.is_available():
        _device = "cuda"
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        # Log GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU memory: {gpu_memory:.2f} GB")
    else:
        _device = "cpu"
        logger.warning("CUDA is not available. Using CPU for inference.")
        logger.info("For GPU support, ensure you have CUDA-compatible PyTorch installed.")

    return _device


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available.

    Returns:
        Dict containing GPU information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": get_device(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            }
        )

    return info
