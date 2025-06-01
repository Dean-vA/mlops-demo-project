import gc
import logging
import os
import sys
from typing import Any, Optional

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .diarization import diarize_audio, is_diarizer_loaded, load_diarizer
from .gpu_utils import get_gpu_info
from .transcription import (
    add_speaker_tags_to_segments,
    extract_data_from_transcription_result,
    is_model_loaded,
    load_model,
    transcribe_audio,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Parakeet STT API with Speaker Diarization")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_memory_usage(stage: str, logger):
    """Log detailed memory usage information."""
    try:
        # Process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # System memory info
        system_memory = psutil.virtual_memory()

        logger.info(f"=== MEMORY DEBUG [{stage}] ===")
        logger.info(f"Process RSS: {memory_info.rss / 1024**3:.2f} GB")
        logger.info(f"Process VMS: {memory_info.vms / 1024**3:.2f} GB")
        logger.info(f"Process Memory %: {memory_percent:.1f}%")
        logger.info(f"System Memory Used: {system_memory.used / 1024**3:.2f} GB / {system_memory.total / 1024**3:.2f} GB")
        logger.info(f"System Memory %: {system_memory.percent:.1f}%")
        logger.info(f"System Available: {system_memory.available / 1024**3:.2f} GB")

        # Python object counts
        logger.info(f"Python objects in memory: {len(gc.get_objects()):,}")

    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")


def log_data_structure_size(obj: Any, name: str, logger, max_depth: int = 2):
    """Log the approximate size of data structures."""
    try:
        # Get size of object
        size_bytes = sys.getsizeof(obj)
        size_mb = size_bytes / 1024**2

        logger.info(f"=== DATA SIZE [{name}] ===")
        logger.info(f"Base size: {size_mb:.2f} MB")
        logger.info(f"Type: {type(obj)}")

        # Analyze structure if it's a dict
        if isinstance(obj, dict):
            logger.info(f"Dict keys: {list(obj.keys())}")
            for key, value in obj.items():
                if max_depth > 0:
                    value_size = sys.getsizeof(value) / 1024**2
                    logger.info(f"  {key}: {type(value)} - {value_size:.2f} MB")

                    # Drill down into large structures
                    if isinstance(value, (list, dict)) and value_size > 1.0:  # > 1MB
                        if isinstance(value, list):
                            logger.info(f"    List length: {len(value)}")
                            if value and max_depth > 1:
                                sample = value[0] if len(value) > 0 else None
                                if sample:
                                    sample_size = sys.getsizeof(sample) / 1024
                                    logger.info(f"    Sample item: {type(sample)} - {sample_size:.2f} KB")
                        elif isinstance(value, dict):
                            logger.info(f"    Dict keys: {list(value.keys())[:10]}...")  # First 10 keys

    except Exception as e:
        logger.warning(f"Could not analyze data structure {name}: {e}")


# Define a simple endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Parakeet STT API with Speaker Diarization!"}


# Define health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint with GPU status.

    Returns:
        dict: Health status of the API including GPU availability.
    """
    gpu_info = get_gpu_info()
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "diarization_model_loaded": is_diarizer_loaded(),
        "gpu_available": gpu_info["cuda_available"],
        "device": gpu_info["device"],
        "gpu_name": gpu_info.get("gpu_name", "N/A"),
    }


@app.get("/gpu-info")
async def gpu_info():
    """Get detailed GPU information.

    Returns:
        dict: Detailed GPU information including memory usage.
    """
    return get_gpu_info()


# Define transcription endpoint
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    return_timestamps: bool = Form(True),
    chunk_duration_sec: Optional[float] = Form(None),
    overlap_duration_sec: Optional[float] = Form(None),
):
    """Transcribe audio file using NVIDIA's Parakeet TDT model.

    Args:
        file: Audio file to transcribe (.wav or .flac)
        return_timestamps: Whether to include word-level timestamps.
        chunk_duration_sec: Duration in seconds for processing audio in chunks.
        overlap_duration_sec: Overlap duration in seconds between chunks.

    Returns:
        dict: Transcription result with text and optional timestamps.
    """
    try:
        # Check file extension
        if not file.filename.lower().endswith((".wav", ".flac")):
            raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")

        # Read file content
        file_content = await file.read()

        # Call transcription function
        result = transcribe_audio(
            file_content,
            file.filename,
            return_timestamps=return_timestamps,
            chunk_duration_sec=chunk_duration_sec,
            overlap_duration_sec=overlap_duration_sec,
        )

        return result

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Define speaker diarization endpoint
@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    return_word_timestamps: bool = Form(True),
):
    """Perform speaker diarization on audio file using NVIDIA's NeMo models.

    Args:
        file: Audio file to diarize (.wav or .flac)
        num_speakers: Number of speakers (if known). If None, auto-detect.
        return_word_timestamps: Whether to include word-level timestamps.

    Returns:
        dict: Diarization result with speaker segments and timing information.
    """
    try:
        # Check file extension
        if not file.filename.lower().endswith((".wav", ".flac")):
            raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")

        # Validate num_speakers if provided
        if num_speakers is not None and (num_speakers < 1 or num_speakers > 20):
            raise HTTPException(status_code=400, detail="Number of speakers must be between 1 and 20")

        # Read file content
        file_content = await file.read()

        # Call diarization function
        result = diarize_audio(
            file_content,
            file.filename,
            num_speakers=num_speakers,
            return_word_timestamps=return_word_timestamps,
        )

        return result

    except Exception as e:
        logger.error(f"Diarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Define combined transcription + diarization endpoint
@app.post("/transcribe_and_diarize")
async def transcribe_and_diarize(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    return_timestamps: bool = Form(True),
):
    """Perform both transcription and speaker diarization on audio file.
    Now includes speaker tags on each transcription segment!

    Args:
        file: Audio file to process (.wav or .flac)
        num_speakers: Number of speakers (if known). If None, auto-detect.
        return_timestamps: Whether to include word-level timestamps.

    Returns:
        dict: Combined transcription and diarization results with speaker-tagged segments.
    """

    try:
        # Check file extension
        if not file.filename.lower().endswith((".wav", ".flac")):
            raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")

        # Validate num_speakers if provided
        if num_speakers is not None and (num_speakers < 1 or num_speakers > 20):
            raise HTTPException(status_code=400, detail="Number of speakers must be between 1 and 20")

        # Read file content
        file_content = await file.read()

        # Run transcription
        logger.info("Starting transcription...")
        transcription_result = transcribe_audio(
            file_content,
            file.filename,
            return_timestamps=return_timestamps,
        )

        # Run diarization
        logger.info("Starting diarization...")
        diarization_result = diarize_audio(
            file_content,
            file.filename,
            num_speakers=num_speakers,
            return_word_timestamps=return_timestamps,
        )

        # Add speaker tags to transcription segments
        logger.info("Adding speaker tags to transcription segments...")
        enhanced_transcription = add_speaker_tags_to_segments(transcription_result, diarization_result)

        # Convert ALL NeMo objects to clean dictionaries
        logger.info("Converting complete transcription to clean format...")
        clean_transcription = extract_data_from_transcription_result(enhanced_transcription)

        # Get processing times from original results
        trans_processing_time = transcription_result.get("processing_time_sec", 0.0) or 0.0
        diar_processing_time = diarization_result.get("processing_time_sec", 0.0) or 0.0
        combined_processing_time = trans_processing_time + diar_processing_time

        # Ensure processing time is preserved in clean result
        clean_transcription["processing_time_sec"] = trans_processing_time

        combined_result = {
            "transcription": clean_transcription,  # Now completely clean!
            "diarization": diarization_result,
            "combined_processing_time": combined_processing_time,
        }

        # Cleanup
        del transcription_result, diarization_result, enhanced_transcription
        gc.collect()

        return combined_result

    except Exception as e:
        logger.error(f"Combined transcription and diarization error: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    try:
        # Log GPU status
        gpu_info = get_gpu_info()
        if gpu_info["cuda_available"]:
            logger.info(f"GPU detected: {gpu_info['gpu_name']}")
            logger.info(f"GPU memory: {gpu_info['gpu_memory_total']:.2f} GB")
        else:
            logger.warning("No GPU detected - will run on CPU (slower performance)")

        # Pre-load the model to speed up first request
        load_model()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        logger.info("Model will be loaded on first request")

    try:
        # Pre-load the diarization model to speed up first request
        load_diarizer()
        logger.info("Diarization model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load diarization model: {e}")
        logger.info("Diarization model will be loaded on first request")
