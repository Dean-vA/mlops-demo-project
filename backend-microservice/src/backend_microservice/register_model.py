"""
Model registration script for D&D LoRA models
Registers models in Azure ML Model Registry based on evaluation criteria
"""

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import mlflow
from azure_utils import register_model as azure_register_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_results(evaluation_path: str) -> Dict[str, Any]:
    """Load evaluation results from the evaluation step."""
    eval_dir = Path(evaluation_path)
    summary_file = eval_dir / "evaluation_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Evaluation summary not found at {summary_file}")

    with open(summary_file, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded evaluation results: BLEU score = {results['avg_bleu_score']:.4f}")
    return results


def should_register_model(evaluation_results: Dict[str, Any], min_score: float) -> bool:
    """Check if model meets registration criteria."""
    avg_bleu = evaluation_results.get("avg_bleu_score", 0.0)

    logger.info("Registration criteria check:")
    logger.info(f"  Average BLEU score: {avg_bleu:.4f}")
    logger.info(f"  Minimum threshold: {min_score:.4f}")

    meets_criteria = avg_bleu >= min_score
    logger.info(f"  Decision: {'✅ REGISTER' if meets_criteria else '❌ SKIP'}")

    return meets_criteria


def create_model_description(evaluation_results: Dict[str, Any]) -> str:
    """Create model description with evaluation results."""
    return f"""# D&D LoRA Summarization Model

Fine-tuned LoRA model based on Llama-3.2-1B-Instruct for D&D session summarization.

## Performance
- Average BLEU Score: {evaluation_results['avg_bleu_score']:.4f}
- Evaluation Samples: {evaluation_results['num_samples']}
- Score Range: {evaluation_results['min_bleu_score']:.4f} - {evaluation_results['max_bleu_score']:.4f}

## Model Details
- Base Model: meta-llama/Llama-3.2-1B-Instruct
- Method: LoRA (Low-Rank Adaptation)
- Domain: D&D session transcripts
- Task: Text summarization

## Usage
Generates detailed D&D session summaries including story progression, character moments, combat encounters, and plot hooks.
"""


def register_model_with_azure(model_path: str, model_name: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Register model in Azure ML."""
    description = create_model_description(evaluation_results)

    tags = {
        "framework": "pytorch",
        "model_type": "lora",
        "base_model": "llama-3.2-1b-instruct",
        "task": "text-summarization",
        "domain": "dnd",
        "avg_bleu_score": f"{evaluation_results['avg_bleu_score']:.4f}",
    }

    properties = {
        "avg_bleu_score": evaluation_results["avg_bleu_score"],
        "min_bleu_score": evaluation_results["min_bleu_score"],
        "max_bleu_score": evaluation_results["max_bleu_score"],
        "num_evaluation_samples": evaluation_results["num_samples"],
    }

    # Try Azure ML registration
    registration_info = azure_register_model(model_path=model_path, model_name=model_name, description=description, tags=tags, properties=properties)

    # Add average BLEU score to registration info
    if registration_info and "avg_bleu_score" not in registration_info:
        registration_info["avg_bleu_score"] = evaluation_results["avg_bleu_score"]

    if registration_info:
        logger.info(f"✅ Model registered: {registration_info['model_name']} v{registration_info['model_version']}")
        return registration_info
    else:
        # Fallback to simulation
        logger.warning("Azure ML registration failed, creating simulation")
        return {
            "status": "simulated",
            "model_name": model_name,
            "model_version": "1",
            "model_id": f"simulated-{uuid.uuid4().hex[:8]}",
            "registration_time": datetime.now().isoformat(),
            "avg_bleu_score": evaluation_results["avg_bleu_score"],
            "note": "Simulated - Azure ML unavailable",
        }


def save_registration_info(registration_info: Dict[str, Any], output_path: str):
    """Save registration information."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    info_file = output_dir / "registration_info.json"
    with open(info_file, "w") as f:
        json.dump(registration_info, f, indent=2)

    # Save report
    report_file = output_dir / "registration_report.md"
    with open(report_file, "w") as f:
        f.write("# Model Registration Report\n\n")
        f.write(f"**Status**: {registration_info['status'].upper()}\n")
        f.write(f"**Model**: {registration_info['model_name']} v{registration_info['model_version']}\n")
        f.write(f"**BLEU Score**: {registration_info['avg_bleu_score']:.4f}\n")
        f.write(f"**Time**: {registration_info['registration_time']}\n")

        if registration_info["status"] == "success":
            f.write(f"**Model ID**: {registration_info['model_id']}\n")
            f.write("\n✅ Successfully registered in Azure ML Model Registry\n")
        else:
            f.write(f"\n⚠️ {registration_info.get('note', 'Registration simulation')}\n")

    logger.info(f"Registration info saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Register D&D LoRA model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--evaluation-path", type=str, required=True, help="Path to evaluation results")
    parser.add_argument("--model-name", type=str, default="dnd-lora-model", help="Model name")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum BLEU score")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    logger.info("Starting model registration...")
    logger.info(f"Model: {args.model_name}, Min BLEU: {args.min_score}")

    try:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({"model_name": args.model_name, "min_score_threshold": args.min_score})

            # Load evaluation results
            evaluation_results = load_evaluation_results(args.evaluation_path)

            # Check registration criteria
            should_register = should_register_model(evaluation_results, args.min_score)

            if should_register:
                # Register model
                registration_info = register_model_with_azure(args.model_path, args.model_name, evaluation_results)

                mlflow.log_metrics({"registration_success": 1, "registered_bleu_score": evaluation_results["avg_bleu_score"]})
                mlflow.log_param("registration_status", "success")

            else:
                # Skip registration
                registration_info = {
                    "status": "skipped",
                    "reason": f"BLEU {evaluation_results['avg_bleu_score']:.4f} < {args.min_score:.4f}",
                    "model_name": args.model_name,
                    "model_version": "N/A",
                    "avg_bleu_score": evaluation_results["avg_bleu_score"],
                    "registration_time": "N/A",
                }

                mlflow.log_metrics({"registration_success": 0})
                mlflow.log_param("registration_status", "skipped")

                logger.info("❌ Registration skipped - below threshold")

            # Save results
            save_registration_info(registration_info, args.output_path)
            logger.info("✅ Registration process completed!")

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise


if __name__ == "__main__":
    main()
