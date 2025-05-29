import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .gpu_utils import get_gpu_info
from .transcription import is_model_loaded, load_model, transcribe_audio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Parakeet STT API")

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
    return {"message": "Welcome to the Parakeet STT API!"}


# Define health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        dict: Health status of the API.
    """
    return {"status": "healthy", "model_loaded": is_model_loaded()}


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
