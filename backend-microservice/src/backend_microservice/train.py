# Check if torch is installed and has GPU support
import argparse
import logging

# MLflow imports
import mlflow
import mlflow.pytorch
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, tokenizer, dataset, output_dir: str, num_epochs: int = 10):
    """Train the LoRA model."""
    logger.info("Starting training process...")

    # Apply LoRA to the model
    model = apply_lora_to_model(model)

    # Get model info
    model_info = get_model_info(model)
    logger.info(f"Model info: {model_info}")

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, output_dir, num_epochs)

    logger.info(f"Training {len(dataset)} samples for {num_epochs} epochs")

    # Start training
    logger.info("Starting training...")
    training_output = trainer.train()

    # Manually log final training metrics
    mlflow.log_metrics(
        {
            "final_train_loss": training_output.training_loss,
            "train_runtime": training_output.metrics.get("train_runtime", 0),
            "train_samples_per_second": training_output.metrics.get("train_samples_per_second", 0),
            "total_parameters": model_info.get("total_parameters", 0),
            "trainable_parameters": model_info.get("trainable_parameters", 0),
        }
    )

    logger.info("Training completed!")
    logger.info(f"Final training loss: {training_output.training_loss:.4f}")

    return trainer, training_output


def train(use_uri: bool, data_path: str, model_path: str, num_epochs: int = 10):
    """
    Main training function.

    Args:
        use_uri (bool): Whether to load data from URI (Azure ML data asset).
        data_path (str): Path to the dataset.
        model_path (str): Path to save the trained model.
        num_epochs (int): Number of training epochs.
    """
    # Enable MLflow autolog with limited parameters to avoid Azure ML limits
    mlflow.pytorch.autolog(
        log_models=False,  # Disable model logging to avoid size issues
        log_datasets=False,  # Disable dataset logging to reduce parameters
        disable=False,  # Enable autolog
        exclusive=False,  # Allow manual logging too
        disable_for_unsupported_versions=True,  # Skip if version incompatible
        silent=True,  # Reduce noise from version warnings
    )

    # Start MLflow run
    with mlflow.start_run():
        logger.info("MLflow autolog enabled - automatic tracking active")

        # Check GPU availability
        device = get_device()
        logger.info(f"Using device: {device}")

        # Log only essential parameters to stay under Azure ML 200 parameter limit
        mlflow.log_params(
            {
                "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                "num_epochs": num_epochs,
                "data_path": data_path.split("/")[-1] if "/" in data_path else data_path,  # Just filename
                "device_type": str(device),
            }
        )

        # Load and prepare data
        if use_uri:
            logger.info("Loading data from URI (Azure ML data asset)")
            df = load_data_from_uri(data_path)
            training_df = prepare_training_dataframe(df)
        else:
            raise NotImplementedError("Local data loading not implemented yet")

        if len(training_df) == 0:
            raise ValueError("No training data found")
        else:
            logger.info(f"Loaded {len(training_df)} training samples")
            logger.info(f"Columns: {list(training_df.columns)}")

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()

        # Prepare dataset directly from DataFrame
        dataset = prepare_dataset_from_dataframe(training_df, tokenizer)

        # Train model - autolog captures everything automatically
        trainer, training_output = train_model(model, tokenizer, dataset, model_path, num_epochs)

        # Save model
        save_model(trainer, tokenizer, model_path)

        logger.info("Training pipeline completed successfully!")
        logger.info("Check Azure ML Studio for automatically logged metrics, parameters, and model!")
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
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset (URI for Azure ML data asset)")
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

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    # Run training
    try:
        train(args.use_uri, args.data_path, args.model_path, args.num_epochs)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
