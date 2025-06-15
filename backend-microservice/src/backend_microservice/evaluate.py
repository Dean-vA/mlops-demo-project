"""
Model evaluation script for D&D LoRA models
Evaluates trained models using BLEU scores and generates sample summaries
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd
import torch

# Import your existing modules
from load_data import load_data_from_uri, prepare_training_dataframe
from model import setup_model_and_tokenizer
from peft import PeftModel

# BLEU score calculation
try:
    from sacrebleu import BLEU

    HAS_SACREBLEU = True
except ImportError:
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu

        nltk.download("punkt", quiet=True)
        HAS_SACREBLEU = False
    except ImportError:
        HAS_SACREBLEU = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str, base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Load the trained LoRA model for inference."""
    logger.info(f"Loading trained model from {model_path}")

    # Load base model and tokenizer
    base_model, tokenizer = setup_model_and_tokenizer()

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference

    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_summary(model, tokenizer, transcript: str, max_length: int = 512) -> str:
    """Generate a summary for a given transcript."""

    # Create input prompt (without target summary)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at summarizing Dungeons & Dragons sessions.
Create engaging, detailed summaries that capture the
story progression, character moments, combat encounters, and future plot hooks.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this D&D session transcript in 300-500 words:

{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

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


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate summaries."""
    if HAS_SACREBLEU is None:
        logger.warning("No BLEU calculation library available")
        return 0.0

    if HAS_SACREBLEU:
        # Using sacrebleu
        bleu = BLEU()
        score = bleu.sentence_score(candidate, [reference])
        return score.score / 100.0  # Convert to 0-1 range
    else:
        # Using NLTK
        from nltk.tokenize import word_tokenize

        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        return sentence_bleu([reference_tokens], candidate_tokens)


def evaluate_model(model_path: str, test_data_path: str, num_samples: int = 10) -> Dict[str, Any]:
    """Evaluate the trained model on test data."""
    logger.info(f"Starting model evaluation on {num_samples} samples")

    # Load test data
    if test_data_path.endswith(".csv"):
        test_df = pd.read_csv(test_data_path)
    else:
        # Assume it's a URI
        test_df = load_data_from_uri(test_data_path)

    test_df = prepare_training_dataframe(test_df)
    logger.info(f"Loaded {len(test_df)} test samples")

    # Limit to requested number of samples
    if num_samples > 0:
        test_df = test_df.head(num_samples)

    # Load trained model
    model, tokenizer = load_trained_model(model_path)

    # Evaluation metrics
    bleu_scores = []
    evaluations = []

    logger.info("Starting evaluation...")

    for idx, row in test_df.iterrows():
        logger.info(f"Evaluating sample {idx + 1}/{len(test_df)}")

        transcript = row["input_text"]
        reference_summary = row["target_summary"]

        # Generate summary
        try:
            generated_summary = generate_summary(model, tokenizer, transcript)

            # Calculate BLEU score
            bleu_score = calculate_bleu_score(reference_summary, generated_summary)
            bleu_scores.append(bleu_score)

            # Store detailed evaluation
            evaluation = {
                "sample_id": idx,
                "chunk_no": row.get("chunk_no", idx),
                "reference_summary": reference_summary,
                "generated_summary": generated_summary,
                "bleu_score": bleu_score,
                "reference_length": len(reference_summary.split()),
                "generated_length": len(generated_summary.split()),
            }
            evaluations.append(evaluation)

            logger.info(f"Sample {idx + 1} BLEU score: {bleu_score:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating sample {idx + 1}: {e}")
            bleu_scores.append(0.0)

    # Calculate aggregate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    results = {
        "num_samples": len(test_df),
        "avg_bleu_score": avg_bleu,
        "min_bleu_score": min(bleu_scores) if bleu_scores else 0.0,
        "max_bleu_score": max(bleu_scores) if bleu_scores else 0.0,
        "individual_scores": bleu_scores,
        "detailed_evaluations": evaluations,
        "model_path": model_path,
    }

    logger.info(f"Evaluation complete! Average BLEU score: {avg_bleu:.4f}")
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to output directory."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics
    summary_file = output_dir / "evaluation_summary.json"
    summary = {
        "num_samples": results["num_samples"],
        "avg_bleu_score": results["avg_bleu_score"],
        "min_bleu_score": results["min_bleu_score"],
        "max_bleu_score": results["max_bleu_score"],
        "model_path": results["model_path"],
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    detailed_file = output_dir / "detailed_evaluations.json"
    with open(detailed_file, "w") as f:
        json.dump(results["detailed_evaluations"], f, indent=2)

    # Save as CSV for easy viewing
    if results["detailed_evaluations"]:
        df = pd.DataFrame(results["detailed_evaluations"])
        csv_file = output_dir / "evaluation_results.csv"
        df.to_csv(csv_file, index=False)

    # Create a human-readable report
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, "w") as f:
        f.write("# D&D LoRA Model Evaluation Report\n\n")
        f.write(f"**Model Path**: {results['model_path']}\n")
        f.write(f"**Samples Evaluated**: {results['num_samples']}\n")
        f.write(f"**Average BLEU Score**: {results['avg_bleu_score']:.4f}\n")
        f.write(f"**Min BLEU Score**: {results['min_bleu_score']:.4f}\n")
        f.write(f"**Max BLEU Score**: {results['max_bleu_score']:.4f}\n\n")

        f.write("## Sample Evaluations\n\n")
        for eval_data in results["detailed_evaluations"][:3]:  # Show first 3 samples
            f.write(f"### Sample {eval_data['sample_id'] + 1}\n")
            f.write(f"**BLEU Score**: {eval_data['bleu_score']:.4f}\n")
            f.write(f"**Reference Length**: {eval_data['reference_length']} words\n")
            f.write(f"**Generated Length**: {eval_data['generated_length']} words\n\n")
            f.write("**Reference Summary**:\n")
            f.write(f"{eval_data['reference_summary'][:200]}...\n\n")
            f.write("**Generated Summary**:\n")
            f.write(f"{eval_data['generated_summary'][:200]}...\n\n")
            f.write("---\n\n")

    logger.info(f"Evaluation results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate D&D LoRA model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for results")

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    try:
        # Start MLflow run for evaluation tracking
        with mlflow.start_run():
            mlflow.log_params({"model_path": args.model_path, "test_data_path": args.test_data_path, "num_samples": args.num_samples})

            # Run evaluation
            results = evaluate_model(args.model_path, args.test_data_path, args.num_samples)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "avg_bleu_score": results["avg_bleu_score"],
                    "min_bleu_score": results["min_bleu_score"],
                    "max_bleu_score": results["max_bleu_score"],
                    "num_evaluated_samples": results["num_samples"],
                }
            )

            # Save results
            save_evaluation_results(results, args.output_path)

            logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
