"""
D&D Session Summarization using OpenAI GPT-4o with chunking support
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import dotenv
from openai import OpenAI

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OpenAI client
_openai_client = None

# Configuration
CHUNK_WORD_LIMIT = 15000  # 50k words per chunk
CHUNK_OVERLAP_WORDS = 500  # 500 word overlap between chunks
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds


def get_openai_client() -> OpenAI:
    """Get or initialize the OpenAI client."""
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")

    return _openai_client


def is_summarization_available() -> bool:
    """Check if summarization is available (OpenAI API key is set)."""
    return os.getenv("OPENAI_API_KEY") is not None


def extract_transcript_text(segments: List[Dict[str, Any]], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Extract full transcript text from segments for chunking.

    Args:
        segments: List of segment dictionaries
        speaker_names: Optional mapping of speaker IDs to readable names

    Returns:
        Full transcript text with speaker labels
    """

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


def chunk_transcript(transcript_text: str) -> List[Dict[str, Any]]:
    """Split transcript into chunks based on word count.

    Args:
        transcript_text: Full transcript text

    Returns:
        List of chunk dictionaries with text and metadata
    """

    words = transcript_text.split()
    total_words = len(words)

    if total_words <= CHUNK_WORD_LIMIT:
        return [{"chunk_id": 1, "text": transcript_text, "word_count": total_words, "start_word": 0, "end_word": total_words}]

    chunks = []
    chunk_id = 1
    start_idx = 0

    while start_idx < total_words:
        # Calculate end index for this chunk
        end_idx = min(start_idx + CHUNK_WORD_LIMIT, total_words)

        # Extract chunk text
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        chunks.append({"chunk_id": chunk_id, "text": chunk_text, "word_count": len(chunk_words), "start_word": start_idx, "end_word": end_idx})

        # Move start index for next chunk, accounting for overlap
        if end_idx < total_words:  # Not the last chunk
            start_idx = end_idx - CHUNK_OVERLAP_WORDS
        else:
            break

        chunk_id += 1

    logger.info(f"Split transcript into {len(chunks)} chunks (total words: {total_words})")
    return chunks


def create_chunk_summary_prompt(chunk_text: str, chunk_id: int, total_chunks: int) -> str:
    """Create prompt for summarizing a single chunk.

    Args:
        chunk_text: Text content of the chunk
        chunk_id: Current chunk number
        total_chunks: Total number of chunks

    Returns:
        Formatted prompt for GPT-4o
    """

    prompt = f"""You are summarizing part {chunk_id} of {total_chunks} from a D&D session transcript.
    Create a concise summary (200-400 words) focusing on the most important events in this segment.

## Your Task
Create a focused summary that captures the key events, character actions, and story progression in this portion of the session.

## Output Structure
Organize your summary with these sections:

### 🎭 Key Events
The most important things that happened in this segment.

### ⚔️ Combat & Challenges
Any battles, skill challenges, or obstacles faced.
Include exciting dice rolls and both in and out-of-character reactions.

### 🎪 Character Actions
Notable things individual characters did or said.

### 📖 Story & Discoveries
Plot advancement, information learned, or mysteries revealed.

## Writing Guidelines
- Write in past tense, narrative style
- Focus on the most significant moments
- Be concise but engaging
- If this is part of a larger session, focus on this segment's unique contributions

## Transcript Segment:
{chunk_text}

Charater context:
- Carl is the Dungeon Master (DM) and is responsible for the overall story and world.
- Dean plays Thorne Thundersoul a human monk.
- Jason play Captain Lothaire a tiefling swashbuckler rogue.
- Stuart plays Kingsley a construct artificer, with flintlock pistols and a chest mounted blunderbuss.
- Keagan plays Dernick Blackbraid a dwarven ranger.
- Kyle plays Dumbledope the wizard, a human wizard and alcoholic.


## Summary:"""

    return prompt


def create_final_summary_prompt(chunk_summaries: List[str]) -> str:
    """Create prompt for final summary synthesis.

    Args:
        chunk_summaries: List of individual chunk summaries

    Returns:
        Formatted prompt for final synthesis
    """

    combined_summaries = "\n\n---\n\n".join([f"Segment {i+1}:\n{summary}" for i, summary in enumerate(chunk_summaries)])

    prompt = f"""You are creating a final cohesive summary from multiple D&D session segments.
    Combine these segment summaries into one comprehensive session summary (1500-2000 words).

    ## Your Task
    Create a complete session summary that weaves together all the segment summaries into a cohesive narrative.

    ## Output Structure
    Organize your summary into these sections:

    ### 🎭 **Session Highlights**
    A brief overview of the session's most important events.

    ### 📖 **Story Progression**
    What major plot points advanced? What did the party discover or accomplish?

    ### ⚔️ **Combat & Challenges**
    Describe combat encounters, puzzles, or skill challenges. Focus on outcomes and memorable moments.
    Include exciting dice rolls and both in and out-of-character reactions.

    ### 🎪 **Character Moments**
    Highlight individual character actions, roleplay moments, and character development.

    ### 🌍 **World Building & NPCs**
    New locations, NPCs introduced, lore revealed, or world elements explored.

    ### 🎣 **Hooks & Next Steps**
    How did the session end? What questions remain? What might happen next?

    ## Writing Guidelines
    - Create a flowing narrative that connects all segments
    - Eliminate redundancy between segments
    - Maintain chronological flow
    - Focus on the most impactful moments
    - Write in engaging, slightly dramatic fantasy style

    ## Segment Summaries to Combine:
    {combined_summaries}

    ## Final Comprehensive Summary:"""

    return prompt


async def generate_chunk_summary_with_retry(chunk_text: str, chunk_id: int, total_chunks: int, model: str = "gpt-4o") -> tuple[str, dict]:
    """Generate summary for a single chunk with retry logic.

    Args:
        chunk_text: Text content of the chunk
        chunk_id: Current chunk number
        total_chunks: Total number of chunks
        model: OpenAI model to use

    Returns:
        Tuple of (generated summary text, token usage dict)
    """

    import asyncio

    client = get_openai_client()
    prompt = create_chunk_summary_prompt(chunk_text, chunk_id, total_chunks)

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Generating summary for chunk {chunk_id}/{total_chunks} (attempt {attempt + 1})")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert D&D session summarizer creating focused segment summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=600,
            )

            summary = response.choices[0].message.content

            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(f"Successfully generated summary for chunk {chunk_id} (tokens: {token_usage['total_tokens']})")
            return summary, token_usage

        except Exception as e:
            logger.error(f"Error generating chunk {chunk_id} summary (attempt {attempt + 1}): {e}")

            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2**attempt)  # Exponential backoff
                logger.info(f"Retrying chunk {chunk_id} in {delay} seconds...")
                await asyncio.sleep(delay)  # Use asyncio.sleep instead of time.sleep
            else:
                raise Exception(f"Failed to generate summary for chunk {chunk_id} after {MAX_RETRIES} attempts: {e}")


async def generate_final_summary_with_retry(chunk_summaries: List[str], model: str = "gpt-4o") -> tuple[str, dict]:
    """Generate final combined summary with retry logic.

    Args:
        chunk_summaries: List of individual chunk summaries
        model: OpenAI model to use

    Returns:
        Tuple of (final combined summary, token usage dict)
    """

    import asyncio

    client = get_openai_client()
    prompt = create_final_summary_prompt(chunk_summaries)

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Generating final combined summary (attempt {attempt + 1})")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert D&D session summarizer creating comprehensive final summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1200,
            )

            final_summary = response.choices[0].message.content

            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(f"Successfully generated final combined summary (tokens: {token_usage['total_tokens']})")
            return final_summary, token_usage

        except Exception as e:
            logger.error(f"Error generating final summary (attempt {attempt + 1}): {e}")

            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2**attempt)
                logger.info(f"Retrying final summary in {delay} seconds...")
                await asyncio.sleep(delay)  # Use asyncio.sleep instead of time.sleep
            else:
                raise Exception(f"Failed to generate final summary after {MAX_RETRIES} attempts: {e}")


async def generate_session_summary(
    segments: List[Dict[str, Any]],
    speaker_names: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Generate a D&D session summary using chunked approach.

    Args:
        segments: List of transcription segments
        speaker_names: Optional mapping of speaker IDs to readable names
        progress_callback: Optional callback for progress updates
        model: OpenAI model to use

    Returns:
        Dictionary containing the summary and metadata
    """

    if not segments:
        raise ValueError("No segments provided for summarization")

    start_time = time.time()

    def update_progress(message: str, progress: float):
        if progress_callback:
            progress_callback(message, progress)
        logger.info(f"Progress: {message} ({progress:.1%})")

    try:
        update_progress("Extracting transcript text...", 0.0)

        # Extract full transcript
        transcript_text = extract_transcript_text(segments, speaker_names)
        word_count = len(transcript_text.split())

        update_progress("Creating chunks...", 0.1)

        # Create chunks
        chunks = chunk_transcript(transcript_text)
        total_chunks = len(chunks)

        logger.info(f"Processing {total_chunks} chunks for summarization")

        # Generate chunk summaries
        chunk_summaries = []
        chunk_metadata = []

        for i, chunk in enumerate(chunks):
            progress = 0.2 + (0.6 * (i / total_chunks))  # 20-80% for chunk processing
            update_progress(f"Generating summary for chunk {i + 1} of {total_chunks}...", progress)

            try:
                chunk_summary, chunk_tokens = await generate_chunk_summary_with_retry(chunk["text"], chunk["chunk_id"], total_chunks, model)

                chunk_summaries.append(chunk_summary)
                chunk_metadata.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "word_count": chunk["word_count"],
                        "start_word": chunk["start_word"],
                        "end_word": chunk["end_word"],
                        "summary_word_count": len(chunk_summary.split()),
                        "prompt_tokens": chunk_tokens.get("prompt_tokens", 0),
                        "completion_tokens": chunk_tokens.get("completion_tokens", 0),
                        "total_tokens": chunk_tokens.get("total_tokens", 0),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to generate summary for chunk {i + 1}: {e}")
                # Continue with other chunks
                chunk_summaries.append(f"[Summary failed for chunk {i + 1}]")
                chunk_metadata.append({"chunk_id": chunk["chunk_id"], "error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        if not chunk_summaries or all(summary.startswith("[Summary failed") for summary in chunk_summaries):
            raise Exception("Failed to generate any chunk summaries")

        # Generate final combined summary
        update_progress("Creating final combined summary...", 0.8)

        # Filter out failed summaries for final synthesis
        valid_summaries = [s for s in chunk_summaries if not s.startswith("[Summary failed")]

        if not valid_summaries:
            raise Exception("No valid chunk summaries available for final synthesis")

        final_summary, final_tokens = await generate_final_summary_with_retry(valid_summaries, model)

        # Add final summary token usage to metadata
        chunk_metadata.append(
            {
                "chunk_id": "final_synthesis",
                "word_count": sum(len(s.split()) for s in valid_summaries),
                "summary_word_count": len(final_summary.split()),
                "prompt_tokens": final_tokens.get("prompt_tokens", 0),
                "completion_tokens": final_tokens.get("completion_tokens", 0),
                "total_tokens": final_tokens.get("total_tokens", 0),
            }
        )

        update_progress("Parsing final summary...", 0.9)

        # Parse sections
        sections = parse_summary_sections(final_summary)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate final metrics
        final_word_count = len(final_summary.split())

        # Calculate total token usage across all chunks
        total_prompt_tokens = sum(metadata.get("prompt_tokens", 0) for metadata in chunk_metadata)
        total_completion_tokens = sum(metadata.get("completion_tokens", 0) for metadata in chunk_metadata)
        total_tokens_used = total_prompt_tokens + total_completion_tokens

        # Estimate costs (GPT-4o pricing as of 2024)
        # Input: $5.00 per 1M tokens, Output: $15.00 per 1M tokens
        estimated_input_cost = (total_prompt_tokens / 1_000_000) * 5.00
        estimated_output_cost = (total_completion_tokens / 1_000_000) * 15.00
        estimated_total_cost = estimated_input_cost + estimated_output_cost

        result = {
            "summary": final_summary,
            "sections": sections,
            "word_count": final_word_count,
            "original_transcript_words": word_count,
            "chunks_processed": len(chunks),
            "chunks_successful": len(valid_summaries),
            "segment_count": len(segments),
            "generation_time_sec": total_time,
            "model_used": model,
            "chunk_metadata": chunk_metadata,
            # Enhanced token usage stats
            "token_usage": {
                "total_tokens": total_tokens_used,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "tokens_per_chunk": round(total_tokens_used / len(chunks)) if chunks else 0,
                "estimated_cost_usd": {
                    "input": round(estimated_input_cost, 4),
                    "output": round(estimated_output_cost, 4),
                    "total": round(estimated_total_cost, 4),
                },
                "efficiency_metrics": {
                    "tokens_per_word_generated": round(total_tokens_used / final_word_count) if final_word_count > 0 else 0,
                    "compression_ratio": round(word_count / final_word_count, 1) if final_word_count > 0 else 0,
                    "processing_speed_words_per_sec": round(word_count / total_time) if total_time > 0 else 0,
                },
            },
        }

        update_progress("Summary generation complete!", 1.0)

        logger.info(f"Summary generated: {final_word_count} words from {word_count} original words, {total_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Failed to generate session summary: {e}")
        raise


def parse_summary_sections(summary_text: str) -> Dict[str, str]:
    """Parse the summary text into sections for better display.

    Args:
        summary_text: The generated summary text

    Returns:
        Dictionary with section names as keys and content as values
    """

    sections = {
        "Session Highlights": "",
        "Story Progression": "",
        "Combat & Challenges": "",
        "Character Moments": "",
        "World Building & NPCs": "",
        "Hooks & Next Steps": "",
    }

    # Split by section headers
    lines = summary_text.split("\n")
    current_section = None
    current_content = []

    for line in lines:
        line = line.strip()

        # Check if this line is a section header
        if line.startswith("###") and "**" in line:
            # Save previous section
            if current_section and current_content:
                sections[current_section] = "\n".join(current_content).strip()

            # Extract section name
            section_name = line.replace("###", "").replace("*", "").strip()
            # Remove emoji if present
            if " " in section_name:
                parts = section_name.split(" ")
                if len(parts) > 1:
                    section_name = " ".join(parts[1:])

            if section_name in sections:
                current_section = section_name
                current_content = []
            else:
                current_section = None
                current_content = []

        elif current_section and line:  # Don't add empty lines
            current_content.append(line)

    # Save the last section
    if current_section and current_content:
        sections[current_section] = "\n".join(current_content).strip()

    # If parsing failed, put everything in Session Highlights
    if not any(sections.values()):
        sections["Session Highlights"] = summary_text

    return sections
