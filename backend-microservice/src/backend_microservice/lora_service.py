"""
LoRA Model Summarization Service
Handles Azure ML endpoint communication for D&D session summarization
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import dotenv
import httpx

# Load environment variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAService:
    """Service for Azure ML LoRA model summarization."""

    def __init__(self):
        self.endpoint_url = os.getenv("AZURE_ML_ENDPOINT_URL")
        self.api_key = os.getenv("AZURE_ML_API_KEY")
        self.timeout = int(os.getenv("AZURE_ML_TIMEOUT", "120"))

        if not self.endpoint_url:
            raise ValueError("AZURE_ML_ENDPOINT_URL environment variable not set")
        if not self.api_key:
            raise ValueError("AZURE_ML_API_KEY environment variable not set")

        logger.info("LoRA service initialized")

    def extract_transcript_text(self, segments: List[Dict[str, Any]], speaker_names: Optional[Dict[str, str]] = None) -> str:
        """Extract and format transcript text from segments."""
        transcript_parts = []

        for segment in segments:
            text = segment.get("segment", segment.get("text", "")).strip()
            speaker = segment.get("speaker", "Unknown")

            if not text:
                continue

            # Map speaker ID to readable name if provided
            if speaker_names and speaker in speaker_names:
                speaker_display = speaker_names[speaker]
            elif speaker.startswith("speaker_"):
                # Convert speaker_0, speaker_1, etc. to Speaker 1, Speaker 2, etc.
                try:
                    num = speaker.split("_")[1]
                    speaker_display = f"Speaker {int(num) + 1}"
                except (IndexError, ValueError):
                    speaker_display = speaker
            else:
                speaker_display = speaker

            transcript_parts.append(f"{speaker_display}: {text}")

        return "\n".join(transcript_parts)

    def clean_model_response(self, raw_response: str) -> str:
        """Clean up the escaped JSON response from Azure ML endpoint."""
        try:
            # The response comes as escaped JSON string
            # First decode the JSON to get the actual string
            if raw_response.startswith('"') and raw_response.endswith('"'):
                # Remove outer quotes and unescape
                cleaned = json.loads(raw_response)
            else:
                cleaned = raw_response

            # Parse the inner JSON
            if isinstance(cleaned, str):
                try:
                    response_data = json.loads(cleaned)
                    summary = response_data.get("summary", "")
                except json.JSONDecodeError:
                    # If it's not JSON, treat as plain text
                    summary = cleaned
            else:
                summary = str(cleaned)

            # Clean up any remaining escape characters and formatting
            summary = summary.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

            # Remove system prompts and extract just the actual summary
            if "assistant" in summary:
                # Split by "assistant" and take everything after it
                parts = summary.split("assistant")
                if len(parts) > 1:
                    summary = parts[-1].strip()

            # Remove any remaining system/user prompts at the beginning
            lines = summary.split("\n")
            clean_lines = []
            skip_prompts = True

            for line in lines:
                line = line.strip()
                # Skip empty lines and system prompts
                if not line:
                    continue

                # Look for the start of actual content (markdown headers or paragraph text)
                if line.startswith("**") or (
                    not any(keyword in line.lower() for keyword in ["system", "user", "assistant", "summarize this", "you are"]) and len(line) > 20
                ):
                    skip_prompts = False

                if not skip_prompts:
                    clean_lines.append(line)

            if clean_lines:
                summary = "\n".join(clean_lines)

            # If the summary ends abruptly (like with "Elric"), add a note
            summary = summary.strip()
            if summary and not summary.endswith((".", "!", "?", '"')):
                # Find the last complete sentence
                sentences = summary.split(".")
                if len(sentences) > 1:
                    # Keep all but the last incomplete sentence
                    summary = ".".join(sentences[:-1]) + "."
                else:
                    # If no complete sentences, add continuation note
                    summary += "..."

            return summary.strip()

        except Exception as e:
            logger.error(f"Error cleaning model response: {e}")
            return raw_response

    async def call_azure_ml_endpoint(self, transcript: str) -> Dict[str, Any]:
        """Make request to Azure ML endpoint."""
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = {"transcript": transcript}

        logger.info("Calling Azure ML LoRA endpoint...")
        start_time = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.endpoint_url, json=data, headers=headers)
                response.raise_for_status()

                end_time = time.time()
                processing_time = end_time - start_time

                # Get the raw response
                raw_response = response.text
                logger.info(f"Azure ML endpoint responded in {processing_time:.2f} seconds")

                # Clean up the response
                cleaned_summary = self.clean_model_response(raw_response)

                return {
                    "summary": cleaned_summary,
                    "processing_time_sec": processing_time,
                    "model_info": "LoRA fine-tuned Llama-3.2-1B-Instruct",
                    "endpoint_url": self.endpoint_url,
                }

            except httpx.HTTPStatusError as e:
                logger.error(f"Azure ML endpoint HTTP error: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Azure ML endpoint error: {e.response.status_code}")
            except httpx.TimeoutException:
                logger.error("Azure ML endpoint request timed out")
                raise Exception("Azure ML endpoint timeout")
            except Exception as e:
                logger.error(f"Error calling Azure ML endpoint: {e}")
                raise

    async def generate_session_summary(self, segments: List[Dict[str, Any]], speaker_names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate session summary using LoRA model."""
        if not segments:
            raise ValueError("No segments provided for summarization")

        start_time = time.time()

        try:
            # Extract transcript text
            transcript_text = self.extract_transcript_text(segments, speaker_names)
            word_count = len(transcript_text.split())

            logger.info(f"Generating LoRA summary for {word_count} words")

            # Call Azure ML endpoint
            ml_response = await self.call_azure_ml_endpoint(transcript_text)

            # Calculate final metrics
            summary_text = ml_response["summary"]
            summary_word_count = len(summary_text.split())

            end_time = time.time()
            total_time = end_time - start_time

            # Format response similar to GPT summarization
            result = {
                "summary": summary_text,
                "word_count": summary_word_count,
                "original_transcript_words": word_count,
                "segment_count": len(segments),
                "generation_time_sec": total_time,
                "model_used": "lora-llama-3.2-1b",
                "model_info": ml_response["model_info"],
                "endpoint_processing_time": ml_response["processing_time_sec"],
                # Add sections parsing if needed
                "sections": self.parse_summary_sections(summary_text),
                # Cost info (much cheaper than GPT-4o)
                "cost_info": {
                    "model_type": "LoRA",
                    "estimated_cost_usd": 0.0,  # Assuming fixed endpoint cost
                    "note": "Azure ML endpoint - fixed cost per request",
                },
            }

            logger.info(f"LoRA summary generated: {summary_word_count} words from {word_count} original words")
            return result

        except Exception as e:
            logger.error(f"Failed to generate LoRA summary: {e}")
            raise

    def parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        """Parse summary into sections for display."""
        sections = {}

        # If the summary has clear section headers with **, parse them
        if "**" in summary_text and summary_text.count("**") >= 2:
            lines = summary_text.split("\n")
            current_section = None
            current_content = []

            for line in lines:
                line = line.strip()
                if line.startswith("**") and line.endswith("**") and len(line) > 4:
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = "\n".join(current_content).strip()

                    # Start new section
                    current_section = line.replace("*", "").strip()
                    # Remove any session numbers or colons
                    if ":" in current_section:
                        current_section = current_section.split(":", 1)[-1].strip()
                    current_content = []
                elif current_section and line:
                    current_content.append(line)

            # Save last section
            if current_section and current_content:
                sections[current_section] = "\n".join(current_content).strip()

        # If no sections were parsed or only one section, put everything in "Session Summary"
        if not sections or len(sections) <= 1:
            sections = {"Session Summary": summary_text}

        return sections


# Service instance
_lora_service = None


def get_lora_service() -> LoRAService:
    """Get singleton LoRA service instance."""
    global _lora_service
    if _lora_service is None:
        _lora_service = LoRAService()
    return _lora_service


def is_lora_available() -> bool:
    """Check if LoRA service is available."""
    try:
        get_lora_service()
        return True
    except Exception:
        return False
