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

from .audio_utils import get_audio_duration
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
        # Check audio duration
        duration = get_audio_duration(temp_file_path)
        logger.info(f"Audio duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        # Transcribe the audio
        logger.info(f"Transcribing audio file: {filename}")

        start_time = time.time()

        # Apply long audio optimizations if > 8 minutes
        optimization_applied = False
        if duration > 480:  # 8 minutes
            try:
                logger.info("Applying long audio optimizations...")
                _model.change_attention_model("rel_pos_local_attn", [256, 256])
                _model.change_subsampling_conv_chunking_factor(1)
                optimization_applied = True
            except Exception as e:
                logger.warning(f"Could not apply optimizations: {e}")

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

        # Revert optimizations if applied
        if optimization_applied:
            try:
                _model.change_attention_model("rel_pos")
                _model.change_subsampling_conv_chunking_factor(-1)
            except Exception as e:
                logger.warning(f"Could not revert optimizations: {e}")

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
        logger.info("=== SPEAKER ALIGNMENT DEBUG START ===")

        # Log initial memory
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory_gb = process.memory_info().rss / 1024**3
        logger.info(f"Initial memory usage: {initial_memory_gb:.2f} GB")

        # Get diarization segments
        speaker_segments = diarization_result.get("segments", [])
        logger.info(f"Speaker segments count: {len(speaker_segments)}")

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
        logger.info(f"Transcription data type: {type(transcription_data)}")

        # Check memory after initial processing
        current_memory_gb = process.memory_info().rss / 1024**3
        logger.info(f"Memory after initial processing: {current_memory_gb:.2f} GB (Δ {current_memory_gb - initial_memory_gb:.2f} GB)")

        # Check if this is a NeMo Hypothesis object
        if hasattr(transcription_data, "timestamp") and transcription_data.timestamp:
            logger.info("Processing NeMo Hypothesis object with timestamps")

            # Access timestamp data directly from the Hypothesis object
            if hasattr(transcription_data.timestamp, "get"):
                segments = transcription_data.timestamp.get("segment", [])
            else:
                segments = getattr(transcription_data.timestamp, "segment", [])

            logger.info(f"Found {len(segments)} transcription segments to process")

            if segments and len(segments) > 0:
                logger.info(f"Found {len(segments)} transcription segments to align")

                # Process segments in batches to monitor memory
                batch_size = 50
                total_segments = len(segments)

                for batch_start in range(0, total_segments, batch_size):
                    batch_end = min(batch_start + batch_size, total_segments)
                    batch_segments = segments[batch_start:batch_end]

                    logger.info(f"Processing segment batch {batch_start+1}-{batch_end}/{total_segments}")

                    # Add speaker labels to each segment in this batch
                    for i, segment in enumerate(batch_segments):
                        actual_segment_num = batch_start + i + 1

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

                            # Only log every 100th segment to avoid spam
                            if actual_segment_num % 100 == 0:
                                logger.info(f"Segment {actual_segment_num} assigned to {best_speaker}")
                        else:
                            segment["speaker"] = "unknown"
                            segment["speaker_confidence"] = 0.0
                            if actual_segment_num % 100 == 0:
                                logger.warning(f"No speaker overlap found for segment {actual_segment_num}")

                    # Check memory after each batch
                    if batch_end % 200 == 0:  # Every 200 segments
                        current_memory_gb = process.memory_info().rss / 1024**3
                        logger.info(
                            f"Memory after processing {batch_end} segments: {current_memory_gb:.2f} GB (Δ {current_memory_gb - initial_memory_gb:.2f} GB)"
                        )

                        # Force garbage collection if memory is growing too much
                        if current_memory_gb - initial_memory_gb > 5.0:  # > 5GB growth
                            logger.warning(f"Memory grew by {current_memory_gb - initial_memory_gb:.2f} GB, forcing GC")
                            import gc

                            gc.collect()
                            after_gc_memory_gb = process.memory_info().rss / 1024**3
                            logger.info(f"Memory after GC: {after_gc_memory_gb:.2f} GB")

                logger.info("Successfully processed all transcription segments with speaker alignment")

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


def extract_data_from_transcription_result(transcription_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract clean data from transcription result, completely replacing NeMo Hypothesis objects.

    This function converts the entire transcription result into a clean dictionary
    that can be safely JSON serialized without complex NeMo objects.

    Args:
        transcription_result: Dictionary containing transcription data with NeMo Hypothesis objects

    Returns:
        Clean dictionary with all NeMo objects converted to simple types
    """
    logger.info("=== EXTRACT COMPLETE TRANSCRIPTION DATA ===")

    # Extract basic fields
    processing_time = transcription_result.get("processing_time_sec", 0.0)
    if processing_time is None:
        processing_time = 0.0
    else:
        processing_time = float(processing_time)

    # Initialize clean result
    clean_result = {
        "processing_time_sec": processing_time,
        "text": "",  # Simple text string
        "segments": [],  # Clean segment list
        "timestamps": {"word": [], "segment": []},
    }

    # Extract text and segments from the NeMo Hypothesis objects
    if transcription_result.get("text") and isinstance(transcription_result["text"], list):
        logger.info(f"Processing {len(transcription_result['text'])} text items")

        for text_item in transcription_result["text"]:
            # Extract the main text
            if hasattr(text_item, "text"):
                clean_result["text"] = str(text_item.text)
            elif isinstance(text_item, dict) and "text" in text_item:
                clean_result["text"] = str(text_item["text"])

            # Extract timestamps if available
            timestamp_data = None
            if hasattr(text_item, "timestamp"):
                timestamp_data = text_item.timestamp
            elif isinstance(text_item, dict) and "timestamp" in text_item:
                timestamp_data = text_item["timestamp"]

            if timestamp_data:
                # Extract word timestamps
                word_timestamps = []
                if hasattr(timestamp_data, "word"):
                    word_data = timestamp_data.word
                elif isinstance(timestamp_data, dict) and "word" in timestamp_data:
                    word_data = timestamp_data["word"]
                else:
                    word_data = None

                if word_data:
                    for word_item in word_data:
                        try:
                            if isinstance(word_item, dict):
                                word_timestamps.append(
                                    {"word": str(word_item.get("word", "")), "start": float(word_item.get("start", 0)), "end": float(word_item.get("end", 0))}
                                )
                            else:
                                # Handle NeMo object
                                word_timestamps.append(
                                    {
                                        "word": str(getattr(word_item, "word", "")),
                                        "start": float(getattr(word_item, "start", 0)),
                                        "end": float(getattr(word_item, "end", 0)),
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Error extracting word timestamp: {e}")
                            continue

                clean_result["timestamps"]["word"] = word_timestamps

                # Extract segment timestamps (with speaker info if available)
                segment_timestamps = []
                segments = []

                if hasattr(timestamp_data, "segment"):
                    segment_data = timestamp_data.segment
                elif isinstance(timestamp_data, dict) and "segment" in timestamp_data:
                    segment_data = timestamp_data["segment"]
                else:
                    segment_data = None

                if segment_data:
                    for seg_item in segment_data:
                        try:
                            if isinstance(seg_item, dict):
                                segment_info = {
                                    "start": float(seg_item.get("start", 0)),
                                    "end": float(seg_item.get("end", 0)),
                                    "text": str(seg_item.get("segment", "")),
                                }
                                # Add speaker info if available
                                if "speaker" in seg_item:
                                    segment_info["speaker"] = str(seg_item["speaker"])
                                    segment_info["speaker_confidence"] = float(seg_item.get("speaker_confidence", 0.0))
                            else:
                                # Handle NeMo object
                                segment_info = {
                                    "start": float(getattr(seg_item, "start", 0)),
                                    "end": float(getattr(seg_item, "end", 0)),
                                    "text": str(getattr(seg_item, "segment", "")),
                                }
                                # Add speaker info if available
                                if hasattr(seg_item, "speaker"):
                                    segment_info["speaker"] = str(seg_item.speaker)
                                    segment_info["speaker_confidence"] = float(getattr(seg_item, "speaker_confidence", 0.0))

                            segment_timestamps.append(segment_info)
                            segments.append(segment_info)

                        except Exception as e:
                            logger.warning(f"Error extracting segment timestamp: {e}")
                            continue

                clean_result["timestamps"]["segment"] = segment_timestamps
                clean_result["segments"] = segments

    logger.info(f"Extracted clean transcription with {len(clean_result['segments'])} segments")
    logger.info("=== EXTRACT COMPLETE TRANSCRIPTION DATA END ===")

    return clean_result
