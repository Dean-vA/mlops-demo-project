"""
Functions for audio transcription using NVIDIA's Parakeet TDT model.
"""

import logging
import os
import tempfile
import time
from typing import Any, BinaryIO, Dict, Optional, Union

import nemo.collections.asr as nemo_asr

from .gpu_utils import get_device

# import torch


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
# MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./.model_cache")
_model = None  # Will hold the loaded model instance


def load_model():
    """Load the Parakeet ASR model.

    Returns:
        The loaded model.
    """
    global _model

    if _model is not None:
        return _model

    logger.info(f"Loading model: {MODEL_NAME}")

    # Create cache directory if it doesn't exist
    # os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Get device
    device = get_device()

    # Load the model
    logger.info(f"Using device: {device}")

    _model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=MODEL_NAME,
        # cache_dir=MODEL_CACHE_DIR
    )
    _model = _model.to(device)
    _model.eval()

    # Enable GPU optimizations if on CUDA
    # if device == "cuda":
    #     # Enable mixed precision for faster inference
    #     _model = _model.half()  # Convert to FP16 for faster GPU inference

    #     # Enable cuDNN autotuner for optimal performance
    #     torch.backends.cudnn.benchmark = True

    #     logger.info("GPU optimizations enabled: FP16 precision, cuDNN autotuner")

    # # For long-form audio, we can limit the attention window
    # # This comes at a cost of slight degradation in performance
    # # but helps with memory usage
    # try:
    #     # These values can be adjusted based on memory constraints
    #     _model.change_attention_model("rel_pos_local_attn", [128, 128])
    #     # Enable chunking for subsampling module
    #     _model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
    #     logger.info("Configured model for long-form audio")
    # except Exception as e:
    #     logger.warning(f"Could not configure long-form audio settings: {e}")

    logger.info("Model loaded successfully")

    # Log model info
    total_params = sum(p.numel() for p in _model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    return _model


def transcribe_audio(
    audio_data: Union[bytes, BinaryIO],
    filename: str,
    return_timestamps: bool = True,
    chunk_duration_sec: Optional[float] = None,
    overlap_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Transcribe audio using NVIDIA's Parakeet TDT model.

    Args:
        audio_data: Audio file content.
        filename: Name of the audio file.
        return_timestamps: Whether to include word-level timestamps.
        chunk_duration_sec: Duration in seconds for processing audio in chunks.
        overlap_duration_sec: Overlap duration in seconds between chunks.

    Returns:
        Dict containing transcription results.
    """
    global _model
    # Load model if not already loaded
    # First check if the model is already loaded
    if not is_model_loaded():
        logger.info("Loading model for the first time")
        # Load the model
        _model = load_model()

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        if isinstance(audio_data, bytes):
            temp_file.write(audio_data)
        else:
            temp_file.write(audio_data.read())

        temp_file_path = temp_file.name

    try:
        # Transcribe the audio
        logger.info(f"Transcribing audio file: {filename}")

        start_time = time.time()

        # # Add chunking parameters if provided
        # if chunk_duration_sec is not None:
        #     transcribe_kwargs["chunk_duration"] = chunk_duration_sec
        #     if overlap_duration_sec is not None:
        #         transcribe_kwargs["overlap_duration"] = overlap_duration_sec

        # Perform transcription
        result = _model.transcribe([temp_file_path], timestamps=return_timestamps)

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Transcription completed in {processing_time:.2f} seconds")

        # Process the results
        transcription_result = {
            "text": result.text if hasattr(result, "text") else result,
            "processing_time_sec": processing_time,
        }

        # Add timestamps if available and requested
        if return_timestamps and hasattr(result, "timestamp"):
            transcription_result["timestamps"] = {
                "word": result.timestamp.get("word", []),
                "segment": result.timestamp.get("segment", []),
            }

            # Format segment timestamps for easier consumption
            segments = []
            for stamp in transcription_result["timestamps"]["segment"]:
                segments.append(
                    {
                        "start": stamp["start"],
                        "end": stamp["end"],
                        "text": stamp["segment"],
                    }
                )

            transcription_result["segments"] = segments

        return transcription_result

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def is_model_loaded() -> bool:
    """Check if the model is already loaded.

    Returns:
        bool: Whether the model is loaded.
    """
    return _model is not None
