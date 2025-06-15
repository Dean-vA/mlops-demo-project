# Check if torch is installed and has GPU support
import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt

# MLflow imports
# MLflow imports
import mlflow  # Check if torch is installed and has GPU support
import numpy as np
from gpu_utils import get_device
from load_data import load_data_from_uri, prepare_training_dataframe
from model import (
    apply_lora_to_model,
    create_trainer,
    get_model_info,
    prepare_dataset_from_dataframe,
    save_model,
    setup_model_and_tokenizer,
)

# Suppress PEFT HuggingFace warnings
warnings.filterwarnings("ignore", message="Unable to fetch remote file due to the following error 401 Client Error")
warnings.filterwarnings("ignore", message="Could not find a config file in meta-llama/Llama-3.2-1B-Instruct")
warnings.filterwarnings("ignore", module="peft.utils.other")
warnings.filterwarnings("ignore", module="peft.utils.save_and_load")

# Reduce HuggingFace verbosity
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPlotter:
    """Helper class to collect and plot training metrics."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.epochs = []
        self.steps = []

    def log_step(self, step, epoch, train_loss=None, val_loss=None, learning_rate=None, grad_norm=None):
        """Log a training step."""
        self.steps.append(step)
        self.epochs.append(epoch)

        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

        # Log to MLflow in real-time
        metrics = {"epoch": epoch}
        if train_loss is not None:
            metrics["train_loss"] = train_loss
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        if grad_norm is not None:
            metrics["gradient_norm"] = grad_norm

        mlflow.log_metrics(metrics, step=step)

    def create_training_plots(self):
        """Create and save training plots."""
        if not self.train_losses:
            logger.warning("No training data to plot")
            return

        # Determine if we have validation data
        has_validation = len(self.val_losses) > 0

        # Create figure with subplots
        if has_validation:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        # Plot 1: Training Loss (and validation if available)
        train_epochs = self.epochs[: len(self.train_losses)]
        ax1.plot(train_epochs, self.train_losses, "b-", linewidth=2, alpha=0.8, label="Training")

        if has_validation:
            val_epochs = self.epochs[: len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, "r-", linewidth=2, alpha=0.8, label="Validation")
            ax1.legend()
            ax1.set_title("Training & Validation Loss")
        else:
            ax1.set_title("Training Loss")

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning Rate Schedule
        if self.learning_rates:
            lr_steps = self.steps[: len(self.learning_rates)]
            ax2.plot(lr_steps, self.learning_rates, "g-", linewidth=2, alpha=0.8)
            ax2.set_xlabel("Training Steps")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # Plot 3: Gradient Norms
        if self.grad_norms:
            grad_steps = self.steps[: len(self.grad_norms)]
            if has_validation:
                ax3.plot(grad_steps, self.grad_norms, "purple", linewidth=2, alpha=0.8)
                ax3.set_xlabel("Training Steps")
                ax3.set_ylabel("Gradient Norm")
                ax3.set_title("Gradient Norms")
                ax3.grid(True, alpha=0.3)
            else:
                ax3.plot(grad_steps, self.grad_norms, "purple", linewidth=2, alpha=0.8)
                ax3.set_xlabel("Training Steps")
                ax3.set_ylabel("Gradient Norm")
                ax3.set_title("Gradient Norms")
                ax3.grid(True, alpha=0.3)

        # Plot 4: Loss comparison (only if validation exists)
        if has_validation:
            min_len = min(len(self.train_losses), len(self.val_losses))
            train_subset = self.train_losses[:min_len]
            val_subset = self.val_losses[:min_len]
            epochs_subset = train_epochs[:min_len]

            ax4.plot(epochs_subset, train_subset, "b-", linewidth=2, alpha=0.8, label="Training", marker="o", markersize=4)
            ax4.plot(epochs_subset, val_subset, "r-", linewidth=2, alpha=0.8, label="Validation", marker="s", markersize=4)
            ax4.set_xlabel("Epochs")
            ax4.set_ylabel("Loss")
            ax4.set_title("Training vs Validation Loss")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Add overfitting indicator
            if len(train_subset) > 2 and len(val_subset) > 2:
                train_trend = np.polyfit(range(len(train_subset)), train_subset, 1)[0]
                val_trend = np.polyfit(range(len(val_subset)), val_subset, 1)[0]
                if train_trend < 0 and val_trend > 0:  # Training decreasing, validation increasing
                    ax4.text(
                        0.05,
                        0.95,
                        "Potential Overfitting",
                        transform=ax4.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        verticalalignment="top",
                        fontweight="bold",
                    )

        plt.tight_layout()

        # Save plot
        plot_path = "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training plots saved to {plot_path}")

        # Log plot to MLflow
        mlflow.log_artifact(plot_path, "plots")

        # Also create individual plots for better MLflow visualization
        self._create_individual_plots()

        plt.close()

    def _create_individual_plots(self):
        """Create individual plots for better MLflow display."""

        # Loss plot with validation if available
        plt.figure(figsize=(12, 8))
        train_epochs = self.epochs[: len(self.train_losses)]
        plt.plot(train_epochs, self.train_losses, "b-", linewidth=3, marker="o", markersize=6, alpha=0.8, label="Training Loss")

        if len(self.val_losses) > 0:
            val_epochs = self.epochs[: len(self.val_losses)]
            plt.plot(val_epochs, self.val_losses, "r-", linewidth=3, marker="s", markersize=6, alpha=0.8, label="Validation Loss")

            # Add trend lines
            if len(self.train_losses) > 1:
                train_trend = np.polyfit(train_epochs, self.train_losses, 1)
                train_p = np.poly1d(train_trend)
                plt.plot(train_epochs, train_p(train_epochs), "b--", alpha=0.6, linewidth=2, label=f"Train Trend (slope: {train_trend[0]:.4f})")

            if len(self.val_losses) > 1:
                val_trend = np.polyfit(val_epochs, self.val_losses, 1)
                val_p = np.poly1d(val_trend)
                plt.plot(val_epochs, val_p(val_epochs), "r--", alpha=0.6, linewidth=2, label=f"Val Trend (slope: {val_trend[0]:.4f})")

            plt.title("Training vs Validation Loss", fontsize=16, fontweight="bold")
        else:
            # Add trend line for training only
            if len(self.train_losses) > 1:
                z = np.polyfit(train_epochs, self.train_losses, 1)
                p = np.poly1d(z)
                plt.plot(train_epochs, p(train_epochs), "b--", alpha=0.6, linewidth=2, label=f"Trend (slope: {z[0]:.4f})")
            plt.title("Training Loss Over Time", fontsize=16, fontweight="bold")

        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_plot_path = "loss_comparison.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(loss_plot_path, "plots")
        plt.close()

        # Learning rate plot (if data available)
        if self.learning_rates:
            plt.figure(figsize=(10, 6))
            lr_steps = self.steps[: len(self.learning_rates)]
            plt.plot(lr_steps, self.learning_rates, "g-", linewidth=3, alpha=0.8)
            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
            plt.tight_layout()
            lr_plot_path = "learning_rate_plot.png"
            plt.savefig(lr_plot_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(lr_plot_path, "plots")
            plt.close()


def parse_training_logs(trainer):
    """Parse training logs from trainer to extract metrics."""
    plotter = TrainingPlotter()

    # Get logs from trainer's log history
    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        step = 0
        for log_entry in trainer.state.log_history:
            epoch = log_entry.get("epoch", 0)
            train_loss = log_entry.get("loss", None)
            val_loss = log_entry.get("eval_loss", None)
            lr = log_entry.get("learning_rate", None)
            grad_norm = log_entry.get("grad_norm", None)

            # Only log if we have at least one metric
            if any(x is not None for x in [train_loss, val_loss, lr, grad_norm]):
                plotter.log_step(step, epoch, train_loss, val_loss, lr, grad_norm)
                step += 1

    return plotter


def train_model(model, tokenizer, train_dataset, val_dataset, output_dir: str, num_epochs: int = 10):
    """Train the LoRA model."""
    logger.info("Starting training process...")

    # Apply LoRA to the model
    model = apply_lora_to_model(model)

    # Get model info
    model_info = get_model_info(model)
    logger.info(f"Model info: {model_info}")

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir, num_epochs)

    logger.info(f"Training {len(train_dataset)} samples for {num_epochs} epochs")
    if val_dataset:
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Start training
    logger.info("Starting training...")
    training_output = trainer.train()

    # Create and log training plots
    logger.info("Creating training visualizations...")
    plotter = parse_training_logs(trainer)
    plotter.create_training_plots()

    # Manually log final training metrics
    final_metrics = {
        "final_train_loss": training_output.training_loss,
        "train_runtime": training_output.metrics.get("train_runtime", 0),
        "train_samples_per_second": training_output.metrics.get("train_samples_per_second", 0),
        "total_parameters": model_info.get("total_parameters", 0),
        "trainable_parameters": model_info.get("trainable_parameters", 0),
    }

    # Add validation metrics if available
    if val_dataset and hasattr(trainer.state, "log_history"):
        # Get the last evaluation metrics
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        if eval_logs:
            final_metrics["final_val_loss"] = eval_logs[-1]["eval_loss"]

    mlflow.log_metrics(final_metrics)

    logger.info("Training completed!")
    logger.info(f"Final training loss: {training_output.training_loss:.4f}")

    return trainer, training_output


def train(use_uri: bool, data_path: str, val_data_path: str, model_path: str, num_epochs: int = 10, validation_split: float = 0.0):
    """
    Main training function.

    Args:
        use_uri (bool): Whether to load data from URI (Azure ML data asset).
        data_path (str): Path to the training dataset.
        val_data_path (str): Path to the validation dataset (optional).
        model_path (str): Path to save the trained model.
        num_epochs (int): Number of training epochs.
        validation_split (float): Fraction of training data to use for validation.
    """

    # Check GPU availability first
    device = get_device()
    logger.info(f"Using device: {device}")

    # Start MLflow run with minimal manual logging
    with mlflow.start_run():
        logger.info("MLflow manual tracking enabled with plots")

        # Log only essential parameters (well under 200 limit)
        params = {
            "model_name": "Llama-3.2-1B-Instruct",
            "num_epochs": num_epochs,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "device": str(device),
            "validation_split": validation_split,
        }

        if val_data_path:
            params["has_validation_data"] = True

        mlflow.log_params(params)

        # Load and prepare training data
        if use_uri:
            logger.info("Loading training data from URI (Azure ML data asset)")
            train_df = load_data_from_uri(data_path)
            train_df = prepare_training_dataframe(train_df)
        else:
            raise NotImplementedError("Local data loading not implemented yet")

        if len(train_df) == 0:
            raise ValueError("No training data found")

        # Handle validation data
        val_df = None
        if val_data_path:
            # Load separate validation file
            logger.info("Loading validation data from separate file")
            if use_uri:
                val_df = load_data_from_uri(val_data_path)
                val_df = prepare_training_dataframe(val_df)
            else:
                raise NotImplementedError("Local validation data loading not implemented yet")

        elif validation_split > 0.0:
            # Split training data
            logger.info(f"Splitting training data: {validation_split:.1%} for validation")
            split_idx = int(len(train_df) * (1 - validation_split))
            val_df = train_df[split_idx:].reset_index(drop=True)
            train_df = train_df[:split_idx].reset_index(drop=True)

        logger.info(f"Training samples: {len(train_df)}")
        if val_df is not None:
            logger.info(f"Validation samples: {len(val_df)}")

        # Log data info
        mlflow.log_params({"training_samples": len(train_df), "validation_samples": len(val_df) if val_df is not None else 0})

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()

        # Prepare datasets
        train_dataset = prepare_dataset_from_dataframe(train_df, tokenizer)
        val_dataset = None
        if val_df is not None:
            logger.info("Preparing validation dataset...")
            val_dataset = prepare_dataset_from_dataframe(val_df, tokenizer)

        # Train model
        trainer, training_output = train_model(model, tokenizer, train_dataset, val_dataset, model_path, num_epochs)

        # Save model
        save_model(trainer, tokenizer, model_path)

        logger.info("Training pipeline completed successfully!")
        logger.info("Check Azure ML Studio for logged metrics, parameters, and plots!")
        return trainer, training_output


if __name__ == "__main__":
    # Check device at startup
    device = get_device()
    logger.info(f"Startup device check: {device}")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train LoRA model for D&D session summarization")
    parser.add_argument(
        "--use-uri",
        action="store_true",
        help="Load data from URI (Azure ML data asset)",
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset (URI for Azure ML data asset)")
    parser.add_argument("--val-data-path", type=str, help="Path to the validation dataset (optional)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/dnd_lora_model",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Fraction of training data to use for validation (0.0-1.0). Only used if --val-data-path not provided.",
    )

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    # Run training
    try:
        train(args.use_uri, args.data_path, args.val_data_path, args.model_path, args.num_epochs, args.validation_split)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
