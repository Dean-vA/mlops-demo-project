"""
Real speaker diarization using NVIDIA's NeMo models with default configurations.
"""

import json
import logging
import os
import tempfile
import time
import urllib.request
from typing import Any, BinaryIO, Dict, List, Optional, Union

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for diarization models
_diarizer_model = None  # Will hold the loaded diarization model
_config_loaded = False


def download_default_config() -> str:
    """Download default NeMo diarization configuration."""
    # Use the general meeting config as it's most versatile
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml"

    # Create temp file for config
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        config_path = f.name

    try:
        logger.info("Downloading default NeMo diarization config...")
        urllib.request.urlretrieve(config_url, config_path)
        logger.info(f"Downloaded config to: {config_path}")
        return config_path
    except Exception as e:
        logger.error(f"Failed to download config: {e}")
        raise


def load_diarizer():
    """Load the speaker diarization model using default NeMo config."""
    global _diarizer_model, _config_loaded

    if _diarizer_model is not None:
        return _diarizer_model

    logger.info("Loading NeMo speaker diarization model...")

    try:
        # Download default config if not already done
        if not _config_loaded:
            config_path = download_default_config()
            _config_loaded = True
        else:
            config_path = download_default_config()  # Re-download for safety

        # Load the config
        config = OmegaConf.load(config_path)

        # Fix Docker multiprocessing issues
        config.num_workers = 0  # Disable multiprocessing for Docker compatibility

        # Set required fields that are marked with ???
        with tempfile.TemporaryDirectory():
            # These will be set dynamically for each request
            config.diarizer.manifest_filepath = None  # Set during inference
            config.diarizer.out_dir = None  # Set during inference

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # The config already has good defaults, we just need to make sure
            # the required fields are accessible
            logger.info("Creating ClusteringDiarizer with downloaded config...")

            # Create the model
            _diarizer_model = nemo_asr.models.ClusteringDiarizer(cfg=config)

            logger.info("NeMo speaker diarization model loaded successfully")
            return _diarizer_model

    except Exception as e:
        logger.error(f"Failed to load NeMo diarization model: {e}")
        logger.error(f"Error type: {type(e).__name__}")

        # Try a simpler approach - create minimal config
        logger.info("Trying simplified config approach...")
        return load_simple_diarizer()


def load_simple_diarizer():
    """Load diarizer with a minimal, working configuration."""
    global _diarizer_model

    try:
        # Create minimal config structure based on working examples
        simple_config = OmegaConf.create(
            {
                "sample_rate": 16000,
                "num_workers": 0,  # Use 0 to avoid multiprocessing issues in Docker
                "batch_size": 1,  # Use batch size of 1 for stability
                "diarizer": {
                    "manifest_filepath": None,
                    "out_dir": None,
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": "vad_multilingual_marblenet",
                        "parameters": {
                            "onset": 0.8,
                            "offset": 0.6,
                            "pad_onset": 0.05,
                            "pad_offset": -0.05,
                            "min_duration_on": 0.1,
                            "min_duration_off": 0.1,
                        },
                    },
                    "speaker_embeddings": {
                        "model_path": "titanet_large",
                        "parameters": {"window_length_in_sec": 1.5, "shift_length_in_sec": 0.75, "multiscale_weights": [1.0], "save_embeddings": False},
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": False,
                            "max_num_speakers": 8,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30,
                            "maj_vote_spk_count": False,
                        }
                    },
                },
            }
        )

        logger.info("Creating ClusteringDiarizer with simplified config...")
        _diarizer_model = nemo_asr.models.ClusteringDiarizer(cfg=simple_config)
        logger.info("Simplified NeMo diarization model loaded successfully")
        return _diarizer_model

    except Exception as e:
        logger.error(f"Failed to load simplified diarizer: {e}")
        logger.error(f"Error details: {str(e)}")
        raise


def diarize_audio(audio_data: Union[bytes, BinaryIO], filename: str, num_speakers: Optional[int] = None, return_word_timestamps: bool = True) -> Dict[str, Any]:
    """Perform real speaker diarization on audio using NVIDIA's NeMo models.

    Args:
        audio_data: Audio file content.
        filename: Name of the audio file.
        num_speakers: Number of speakers (if known). If None, auto-detect.
        return_word_timestamps: Whether to include word-level timestamps.

    Returns:
        Dict containing diarization results with speaker labels and segments.
    """
    global _diarizer_model

    # Load model if not already loaded
    if not is_diarizer_loaded():
        logger.info("Loading diarization model for the first time")
        _diarizer_model = load_diarizer()

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save audio to temporary file
        audio_path = os.path.join(temp_dir, filename)
        if isinstance(audio_data, bytes):
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        else:
            with open(audio_path, "wb") as f:
                f.write(audio_data.read())

        # Create manifest file for NeMo (newline-delimited JSON format)
        manifest_data = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }

        manifest_path = os.path.join(temp_dir, "input_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)
            f.write("\n")  # NeMo expects newline-delimited JSON

        try:
            logger.info(f"Starting speaker diarization for: {filename}")
            start_time = time.time()

            # Update model configuration for this specific inference
            _diarizer_model._diarizer_params.manifest_filepath = manifest_path
            _diarizer_model._diarizer_params.out_dir = temp_dir

            # Set oracle num speakers if provided
            if num_speakers is not None:
                logger.info(f"Using oracle number of speakers: {num_speakers}")
                _diarizer_model._cluster_params.oracle_num_speakers = True
                # Update the manifest with num_speakers info
                manifest_data["num_speakers"] = num_speakers
                with open(manifest_path, "w") as f:
                    json.dump(manifest_data, f)
                    f.write("\n")
            else:
                logger.info("Auto-detecting number of speakers")
                _diarizer_model._cluster_params.oracle_num_speakers = False

            # Run the actual diarization
            logger.info("Running NeMo diarization...")
            try:
                _diarizer_model.diarize()
            except ValueError as e:
                if "silence" in str(e).lower():
                    # Handle case where audio contains only silence
                    logger.warning(f"Audio file contains only silence: {e}")
                    end_time = time.time()
                    processing_time = end_time - start_time

                    # Return empty result for silence-only audio
                    result = {
                        "speakers": {},
                        "num_speakers_detected": 0,
                        "processing_time_sec": processing_time,
                        "segments": [],
                        "model_info": "Real NeMo ClusteringDiarizer with TitaNet + MarbleNet VAD",
                        "note": "No speech detected in audio file (silence only)",
                    }
                    return result
                else:
                    # Re-raise other ValueError exceptions
                    raise

            end_time = time.time()
            processing_time = end_time - start_time

            # Read RTTM output file
            rttm_dir = os.path.join(temp_dir, "pred_rttms")

            if not os.path.exists(rttm_dir):
                raise Exception("No RTTM output directory found")

            rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith(".rttm")]

            if not rttm_files:
                raise Exception("No RTTM output file generated")

            rttm_file = os.path.join(rttm_dir, rttm_files[0])

            # Debug: Log RTTM file contents
            # logger.info(f"Reading RTTM file: {rttm_file}")
            # try:
            #     with open(rttm_file, "r") as f:
            #         rttm_content = f.read()
            #         # logger.info(f"RTTM file contents:\n{rttm_content}")
            # except Exception as debug_e:
            #     logger.warning(f"Could not read RTTM for debugging: {debug_e}")

            speaker_segments = parse_rttm_file(rttm_file)

            logger.info(f"Real diarization completed in {processing_time:.2f} seconds")

            if speaker_segments:
                unique_speakers = len(set(seg["speaker"] for seg in speaker_segments))
                logger.info(f"Detected {unique_speakers} speakers with {len(speaker_segments)} total segments")
            else:
                logger.warning("No speaker segments found in RTTM output")
                unique_speakers = 0

            # Format results
            result = {
                "speakers": format_speaker_segments(speaker_segments),
                "num_speakers_detected": unique_speakers,
                "processing_time_sec": processing_time,
                "segments": speaker_segments,
                "model_info": "Real NeMo ClusteringDiarizer with TitaNet + MarbleNet VAD",
            }

            return result

        except Exception as e:
            logger.error(f"Real diarization failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


def parse_rttm_file(rttm_path: str) -> List[Dict[str, Any]]:
    """Parse RTTM file and extract speaker segments.

    Args:
        rttm_path: Path to RTTM file.

    Returns:
        List of speaker segments with timing and speaker info.
    """
    segments = []

    try:
        with open(rttm_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        try:
                            # RTTM format: SPEAKER <file> 1 <start> <duration> <U> <U> <speaker> <conf> <slat>
                            # The filename can have spaces, so we need to find the numeric fields carefully

                            # Standard RTTM has these fields in order after SPEAKER:
                            # filename, channel, start_time, duration, unk1, unk2, speaker_id, confidence, unk3

                            # Find positions of numeric fields by looking for the pattern:
                            # ... 1 <float> <float> <NA> <NA> speaker_X <NA> <NA>

                            # Look for channel "1" first, then the two floats after it
                            channel_idx = None
                            for i, part in enumerate(parts[1:], 1):  # Skip "SPEAKER"
                                if part == "1":
                                    channel_idx = i
                                    break

                            if channel_idx is None:
                                logger.warning(f"Could not find channel field in RTTM line: {line}")
                                continue

                            # After channel, expect: start_time, duration, <NA>, <NA>, speaker_id, <NA>, <NA>
                            if len(parts) < channel_idx + 6:
                                logger.warning(f"Not enough fields after channel in RTTM line: {line}")
                                continue

                            start_time = float(parts[channel_idx + 1])
                            duration = float(parts[channel_idx + 2])
                            # parts[channel_idx + 3] should be <NA>
                            # parts[channel_idx + 4] should be <NA>
                            speaker = parts[channel_idx + 5]
                            # parts[channel_idx + 6] might be confidence or <NA>

                            end_time = start_time + duration
                            confidence = 1.0  # Default confidence since NeMo outputs <NA>

                            segments.append({"start": start_time, "end": end_time, "duration": duration, "speaker": speaker, "confidence": confidence})

                        except (ValueError, IndexError) as e:
                            logger.warning(f"Skipping malformed RTTM line {line_num}: {line.strip()}")
                            logger.warning(f"Parse error: {e}")
                            continue

    except Exception as e:
        logger.error(f"Error reading RTTM file {rttm_path}: {e}")
        raise

    # Sort segments by start time
    segments.sort(key=lambda x: x["start"])

    logger.info(f"Successfully parsed {len(segments)} segments from RTTM file")
    return segments


def format_speaker_segments(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Format speaker segments grouped by speaker."""
    speakers = {}

    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speakers:
            speakers[speaker] = []

        speakers[speaker].append(
            {"start": segment["start"], "end": segment["end"], "duration": segment["duration"], "confidence": segment.get("confidence", 1.0)}
        )

    return speakers


def is_diarizer_loaded() -> bool:
    """Check if the diarization model is already loaded."""
    return _diarizer_model is not None
