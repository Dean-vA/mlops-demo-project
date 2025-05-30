import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .diarization import diarize_audio, is_diarizer_loaded, load_diarizer
from .gpu_utils import get_gpu_info
from .transcription import is_model_loaded, load_model, transcribe_audio

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
        from .transcription import add_speaker_tags_to_segments

        enhanced_transcription = add_speaker_tags_to_segments(transcription_result, diarization_result)

        # Combine results
        combined_result = {
            "transcription": enhanced_transcription,  # Now includes speaker tags!
            "diarization": diarization_result,
            "combined_processing_time": (transcription_result.get("processing_time_sec", 0) + diarization_result.get("processing_time_sec", 0)),
        }

        return combined_result

    except Exception as e:
        logger.error(f"Combined transcription and diarization error: {str(e)}")
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
