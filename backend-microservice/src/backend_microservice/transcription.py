"""
Functions for audio transcription using NVIDIA's Parakeet TDT model.
"""

import gc
import logging
import os
import tempfile
import time
from typing import Any, BinaryIO, Dict, Optional, Union

import nemo.collections.asr as nemo_asr
import torch

from .gpu_utils import get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
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

    # Get device
    device = get_device()

    # Load the model
    logger.info(f"Using device: {device}")

    _model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=MODEL_NAME,
    )
    _model = _model.to(device)
    _model.eval()

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
        overlap_duration_sec: Optional[float] = None,

    Returns:
        Dict containing transcription results.
    """
    global _model

    # Load model if not already loaded
    if not is_model_loaded():
        logger.info("Loading model for the first time")
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

        # Force garbage collection before transcription
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Perform transcription with memory management
        logger.info(f"Starting transcription with timestamps: {return_timestamps}")

        # Use torch.no_grad() to prevent memory buildup from gradients
        with torch.no_grad():
            result = _model.transcribe([temp_file_path], timestamps=return_timestamps)

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Transcription completed in {processing_time:.2f} seconds")

        # Process the results with careful memory management
        transcription_result = {
            "processing_time_sec": processing_time,
        }

        # Handle different result formats from NeMo
        if hasattr(result, "text"):
            transcription_result["text"] = result.text
        elif isinstance(result, list) and len(result) > 0:
            transcription_result["text"] = result
        else:
            transcription_result["text"] = str(result)

        # Add timestamps if available and requested
        if return_timestamps:
            logger.info("Processing timestamps...")

            # Initialize timestamp containers
            transcription_result["timestamps"] = {"word": [], "segment": []}
            transcription_result["segments"] = []

            # Process timestamps carefully to avoid memory issues
            try:
                if hasattr(result, "timestamp") and result.timestamp:
                    # Extract word timestamps
                    if "word" in result.timestamp:
                        word_timestamps = result.timestamp["word"]
                        if word_timestamps:
                            # Process in smaller batches to avoid memory spike
                            batch_size = 100
                            processed_words = []

                            for i in range(0, len(word_timestamps), batch_size):
                                batch = word_timestamps[i : i + batch_size]
                                processed_words.extend(batch)

                                # Clear intermediate data
                                if i % (batch_size * 10) == 0:  # Every 1000 words
                                    gc.collect()

                            transcription_result["timestamps"]["word"] = processed_words

                    # Extract segment timestamps
                    if "segment" in result.timestamp:
                        segment_timestamps = result.timestamp["segment"]
                        if segment_timestamps:
                            transcription_result["timestamps"]["segment"] = segment_timestamps

                            # Format segment timestamps for easier consumption
                            segments = []
                            for stamp in segment_timestamps:
                                segments.append({"start": stamp.get("start", 0), "end": stamp.get("end", 0), "text": stamp.get("segment", "")})

                            transcription_result["segments"] = segments

                logger.info(f"Processed {len(transcription_result['timestamps']['word'])} word timestamps")
                logger.info(f"Processed {len(transcription_result['segments'])} segments")

            except Exception as e:
                logger.warning(f"Error processing timestamps: {e}")
                # Continue without timestamps rather than crashing
                transcription_result["timestamps"] = {"word": [], "segment": []}
                transcription_result["segments"] = []

        # Force cleanup of the result object to free memory
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Transcription processing complete")
        return transcription_result

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

        # Final cleanup
        gc.collect()


def is_model_loaded() -> bool:
    """Check if the model is already loaded.

    Returns:
        bool: Whether the model is loaded.
    """
    return _model is not None


def add_speaker_tags_to_segments(transcription_result, diarization_result):
    """Add speaker tags to transcription segments based on overlap timing.

    Args:
        transcription_result: Result from transcribe_audio()
        diarization_result: Result from diarize_audio()

    Returns:
        Modified transcription_result with speaker tags added to segments
    """
    try:
        # Get diarization segments
        speaker_segments = diarization_result.get("segments", [])

        if not speaker_segments:
            logger.info("No speaker segments found, marking all as 'unknown'")
            return transcription_result

        logger.info(f"Found {len(speaker_segments)} speaker segments for alignment")

        # Handle the case where transcription_result["text"] contains NeMo Hypothesis objects
        if not transcription_result.get("text") or not isinstance(transcription_result["text"], list):
            logger.warning("No transcription text found or wrong format")
            return transcription_result

        if len(transcription_result["text"]) == 0:
            logger.warning("Empty transcription text list")
            return transcription_result

        transcription_data = transcription_result["text"][0]

        # Check if this is a NeMo Hypothesis object
        if hasattr(transcription_data, "timestamp") and transcription_data.timestamp:
            logger.info("Processing NeMo Hypothesis object with timestamps")

            # Access timestamp data directly from the Hypothesis object
            if hasattr(transcription_data.timestamp, "get"):
                # It's a dictionary
                segments = transcription_data.timestamp.get("segment", [])
            else:
                # It's an object with direct access
                segments = getattr(transcription_data.timestamp, "segment", [])

            if segments and len(segments) > 0:
                logger.info(f"Found {len(segments)} transcription segments to align")

                # Add speaker labels to each segment
                for i, segment in enumerate(segments):
                    trans_start = segment.get("start", 0)
                    trans_end = segment.get("end", 0)
                    trans_duration = trans_end - trans_start

                    # Find which speakers overlap with this transcription segment
                    speaker_overlaps = {}

                    for speaker_seg in speaker_segments:
                        speaker = speaker_seg["speaker"]
                        spk_start = speaker_seg["start"]
                        spk_end = speaker_seg["end"]

                        # Calculate overlap between transcription segment and speaker segment
                        overlap_start = max(trans_start, spk_start)
                        overlap_end = min(trans_end, spk_end)
                        overlap_duration = max(0, overlap_end - overlap_start)

                        if overlap_duration > 0:
                            if speaker not in speaker_overlaps:
                                speaker_overlaps[speaker] = 0
                            speaker_overlaps[speaker] += overlap_duration

                    # Assign speaker with longest overlap time
                    if speaker_overlaps:
                        best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                        segment["speaker"] = best_speaker
                        segment["speaker_confidence"] = speaker_overlaps[best_speaker] / trans_duration if trans_duration > 0 else 0
                        logger.info(f"Segment {i+1} '{segment.get('segment', '')[:50]}...' assigned to {best_speaker}")
                    else:
                        # No overlap found, assign "unknown"
                        segment["speaker"] = "unknown"
                        segment["speaker_confidence"] = 0.0
                        logger.warning(f"No speaker overlap found for segment {i+1}")

                logger.info("Successfully added speaker tags to all transcription segments")

            else:
                logger.warning("No segments found in timestamp object")

        # Check if it's a regular dictionary format (fallback)
        elif isinstance(transcription_data, dict) and transcription_data.get("timestamp", {}).get("segment"):
            logger.info("Processing regular dictionary format with timestamps")

            transcription_segments = transcription_data["timestamp"]["segment"]

            for trans_segment in transcription_segments:
                trans_start = trans_segment.get("start", 0)
                trans_end = trans_segment.get("end", 0)
                trans_duration = trans_end - trans_start

                # Find which speakers overlap with this transcription segment
                speaker_overlaps = {}

                for speaker_seg in speaker_segments:
                    speaker = speaker_seg["speaker"]
                    spk_start = speaker_seg["start"]
                    spk_end = speaker_seg["end"]

                    # Calculate overlap between transcription segment and speaker segment
                    overlap_start = max(trans_start, spk_start)
                    overlap_end = min(trans_end, spk_end)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > 0:
                        if speaker not in speaker_overlaps:
                            speaker_overlaps[speaker] = 0
                        speaker_overlaps[speaker] += overlap_duration

                # Assign speaker with longest overlap time
                if speaker_overlaps:
                    best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                    trans_segment["speaker"] = best_speaker
                    trans_segment["speaker_confidence"] = speaker_overlaps[best_speaker] / trans_duration if trans_duration > 0 else 0
                else:
                    # No overlap found, assign "unknown"
                    trans_segment["speaker"] = "unknown"
                    trans_segment["speaker_confidence"] = 0.0

        else:
            logger.warning(f"Unrecognized transcription data format: {type(transcription_data)}")

        return transcription_result

    except Exception as e:
        logger.error(f"Failed to add speaker tags: {e}")
        logger.exception("Full error details:")
        return transcription_result
