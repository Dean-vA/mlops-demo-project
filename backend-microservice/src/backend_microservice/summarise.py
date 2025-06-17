"""
Simple summarization functions for D&D LoRA model inference
For use in Azure ML endpoint scoring scripts
"""

import logging
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

# Global variables to cache loaded model and tokenizer
_model = None
_tokenizer = None


def load_model(model_path: str, base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> tuple:
    """
    Load the trained LoRA model and tokenizer.

    Args:
        model_path: Path to the LoRA adapter weights
        base_model_name: Base model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        logger.info("Model already loaded, returning cached version")
        return _model, _tokenizer

    logger.info(f"Loading model from {model_path}")

    try:
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        _tokenizer.pad_token = _tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

        # Load LoRA weights and merge
        _model = PeftModel.from_pretrained(base_model, model_path)
        _model = _model.merge_and_unload()
        _model.eval()

        logger.info("Model loaded successfully")
        return _model, _tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_prompt(transcript: str) -> str:
    """
    Create a formatted prompt for the model.

    Args:
        transcript: The D&D transcript to summarize

    Returns:
        Formatted prompt string
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at summarizing Dungeons & Dragons sessions.
Create engaging, detailed summaries that capture the
story progression, character moments, combat encounters, and future plot hooks.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this D&D session transcript in 300-500 words:

{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def generate_summary(transcript: str, model_path: str, max_length: int = 512) -> str:
    """
    Generate a summary for the given transcript.

    Args:
        transcript: The D&D transcript to summarize
        model_path: Path to the LoRA model
        max_length: Maximum length of generated summary

    Returns:
        Generated summary text
    """
    # Load model if not already loaded
    model, tokenizer = load_model(model_path)

    # Create prompt
    prompt = create_prompt(transcript)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False).to(model.device)  # Leave room for generation

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract generated summary
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the generated summary (after the assistant header)
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        summary = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        # Remove any trailing tokens
        if "<|eot_id|>" in summary:
            summary = summary.split("<|eot_id|>")[0].strip()
    else:
        summary = full_response

    return summary


def batch_summarize(transcripts: list, model_path: str, max_length: int = 512) -> list:
    """
    Generate summaries for multiple transcripts.

    Args:
        transcripts: List of D&D transcripts to summarize
        model_path: Path to the LoRA model
        max_length: Maximum length of generated summaries

    Returns:
        List of generated summaries
    """
    summaries = []

    # Load model once for all transcripts
    model, tokenizer = load_model(model_path)

    for i, transcript in enumerate(transcripts):
        logger.info(f"Generating summary {i+1}/{len(transcripts)}")
        try:
            summary = generate_summary(transcript, model_path, max_length)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to generate summary for transcript {i+1}: {e}")
            summaries.append(f"Error: Could not generate summary - {str(e)}")

    return summaries


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary with model information
    """
    global _model, _tokenizer

    if _model is None:
        return {"status": "Model not loaded"}

    try:
        device = next(_model.parameters()).device
        dtype = next(_model.parameters()).dtype

        return {
            "status": "Model loaded",
            "device": str(device),
            "dtype": str(dtype),
            "model_type": "LoRA fine-tuned Llama-3.2-1B-Instruct",
            "task": "D&D session summarization",
        }
    except Exception as e:
        return {"status": "Error getting model info", "error": str(e)}


def cleanup():
    """
    Clean up loaded model to free memory.
    """
    global _model, _tokenizer

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model cleanup completed")


# Simple usage example for testing
if __name__ == "__main__":
    # Example usage
    test_transcript = """
    The party entered the dark dungeon. The rogue checked for traps while the wizard
    cast a light spell. They encountered a group of goblins and had an epic battle.
    The fighter rolled a natural 20! After defeating the enemies, they found a
    treasure chest with gold and magical items.
    """

    model_path = "path/to/your/lora/model"  # Update this path

    try:
        summary = generate_summary(test_transcript, model_path)
        print("Generated Summary:")
        print(summary)

        info = get_model_info()
        print("\nModel Info:")
        print(info)

    except Exception as e:
        print(f"Error: {e}")
