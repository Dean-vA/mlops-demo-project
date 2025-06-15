"""
Model setup and configuration for LoRA fine-tuning
Handles model loading, LoRA configuration, and training setup
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import dotenv
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

# ML Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load environment variables from .env file
dotenv.load_dotenv()


def setup_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and configure the base model and tokenizer.

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {MODEL_NAME}")

    # Get HuggingFace token from environment
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_HUB_TOKEN not found in environment variables")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    # Set longer timeout for huggingface downloads
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")

    # Configure quantization for memory efficiency
    logger.info("Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, token=hf_token
    )

    logger.info(f"Model loaded successfully on: {model.device}")
    return model, tokenizer


def setup_lora_config() -> LoraConfig:
    """
    Configure LoRA for parameter-efficient fine-tuning.

    Returns:
        LoraConfig: Configured LoRA settings
    """
    logger.info("Setting up LoRA configuration...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank - appropriate for small dataset
        lora_alpha=32,  # Scaling parameter
        lora_dropout=0.1,  # Dropout for regularization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target attention and MLP modules
        bias="none",
        use_rslora=False,
    )

    logger.info("LoRA configuration created:")
    logger.info(f"  - Rank (r): {lora_config.r}")
    logger.info(f"  - Alpha: {lora_config.lora_alpha}")
    logger.info(f"  - Dropout: {lora_config.lora_dropout}")
    logger.info(f"  - Target modules: {lora_config.target_modules}")

    return lora_config


def apply_lora_to_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Apply LoRA configuration to the model.

    Args:
        model: Base model to apply LoRA to

    Returns:
        Model with LoRA applied
    """
    logger.info("Applying LoRA to model...")

    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    model.print_trainable_parameters()

    return model


def create_training_prompt(input_text: str, target_summary: str) -> str:
    """
    Create a formatted prompt for training using Llama's chat template.

    Args:
        input_text: The D&D transcript to summarize
        target_summary: The target summary for training

    Returns:
        Formatted prompt string
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at summarizing Dungeons & Dragons sessions.
Create engaging, detailed summaries that capture the
story progression, character moments, combat encounters, and future plot hooks.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this D&D session transcript in 300-500 words:

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_summary}<|eot_id|>"""
    return prompt


def tokenize_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int = 2048) -> Dict:
    """
    Tokenize the training examples.

    Args:
        examples: Batch of examples with 'input_text' and 'target_summary'
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    # Create full prompts - handle both single examples and batches
    if isinstance(examples["input_text"], list):
        prompts = [create_training_prompt(inp, target) for inp, target in zip(examples["input_text"], examples["target_summary"])]
    else:
        prompts = [create_training_prompt(examples["input_text"], examples["target_summary"])]

    # Tokenize with proper parameters
    tokenized = tokenizer(prompts, truncation=True, padding=True, max_length=max_length, return_tensors=None)  # Important: return Python lists, not tensors

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def prepare_dataset_from_dataframe(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    """
    Convert DataFrame directly to HuggingFace Dataset format and tokenize.

    Args:
        df: DataFrame with 'input_text' and 'target_summary' columns
        tokenizer: The tokenizer to use

    Returns:
        Dataset: Tokenized dataset ready for training
    """
    logger.info("Preparing dataset from DataFrame...")

    # Validate required columns
    required_cols = ["input_text", "target_summary"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Ensure data types are correct and clean
    df_clean = df.copy()
    df_clean["input_text"] = df_clean["input_text"].astype(str).str.strip()
    df_clean["target_summary"] = df_clean["target_summary"].astype(str).str.strip()

    # Remove any rows with empty strings
    df_clean = df_clean[
        (df_clean["input_text"] != "") & (df_clean["target_summary"] != "") & (df_clean["input_text"].notna()) & (df_clean["target_summary"].notna())
    ].reset_index(drop=True)

    logger.info(f"Cleaned dataset: {len(df_clean)} examples")

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df_clean[["input_text", "target_summary"]])
    logger.info(f"Created dataset with {len(dataset)} examples")

    # Debug: Check data structure before tokenization
    logger.info("Sample data before tokenization:")
    logger.info(f"Input type: {type(dataset[0]['input_text'])}")
    logger.info(f"Target type: {type(dataset[0]['target_summary'])}")
    logger.info(f"Input preview: {dataset[0]['input_text'][:100]}...")

    # Tokenize dataset with error handling
    logger.info("Tokenizing dataset...")
    try:
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            batch_size=1,  # Process one at a time to debug issues
            remove_columns=dataset.column_names,  # Remove original columns
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        # Try with single example to debug
        logger.info("Attempting single example tokenization for debugging...")
        try:
            single_example = {"input_text": [dataset[0]["input_text"]], "target_summary": [dataset[0]["target_summary"]]}
            result = tokenize_function(single_example, tokenizer)
            logger.info(f"Single tokenization successful. Result keys: {result.keys()}")
            logger.info(f"Input_ids type: {type(result['input_ids'])}")
            logger.info(f"Input_ids length: {len(result['input_ids'])}")
        except Exception as debug_e:
            logger.error(f"Single example tokenization also failed: {debug_e}")
        raise

    # Log dataset statistics
    if len(tokenized_dataset) > 0:
        avg_length = np.mean([len(x) for x in tokenized_dataset["input_ids"]])
        max_length = max([len(x) for x in tokenized_dataset["input_ids"]])
        min_length = min([len(x) for x in tokenized_dataset["input_ids"]])

        logger.info("Dataset tokenization complete:")
        logger.info(f"  - Total examples: {len(tokenized_dataset)}")
        logger.info(f"  - Average sequence length: {avg_length:.0f}")
        logger.info(f"  - Max sequence length: {max_length}")
        logger.info(f"  - Min sequence length: {min_length}")

    return tokenized_dataset


def setup_training_args(output_dir: str, num_epochs: int = 10, has_validation: bool = False) -> TrainingArguments:
    """
    Configure training arguments with minimal parameters to avoid Azure ML limits.

    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        has_validation: Whether validation data is provided

    Returns:
        TrainingArguments: Configured training arguments
    """
    logger.info("Setting up training arguments...")

    # Configure evaluation strategy based on validation data
    eval_strategy = "epoch" if has_validation else "no"
    eval_steps = 1 if has_validation else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,  # Reduced to save space
        warmup_steps=2,
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[],  # IMPORTANT: Empty list disables all integrations including MLflow
        push_to_hub=False,
        # Validation settings - using eval_strategy for compatibility with older transformers
        eval_strategy=eval_strategy,  # Changed from evaluation_strategy
        eval_steps=eval_steps,
        per_device_eval_batch_size=1,
        # Disable MLflow integration at the trainer level
        disable_tqdm=False,
        load_best_model_at_end=has_validation,  # Load best model if we have validation
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
    )

    logger.info("Training arguments configured:")
    logger.info(f"  - Epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    logger.info(f"  - Evaluation strategy: {training_args.eval_strategy}")
    logger.info("  - MLflow auto-integration: DISABLED")

    return training_args


def create_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForLanguageModeling:
    """
    Create data collator for language modeling.

    Args:
        tokenizer: The tokenizer to use

    Returns:
        DataCollatorForLanguageModeling: Configured data collator
    """
    logger.info("Creating data collator...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for better performance
        return_tensors="pt",  # Ensure tensors are returned
    )

    return data_collator


def create_trainer(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, train_dataset: Dataset, val_dataset: Dataset, output_dir: str, num_epochs: int = 10
) -> Trainer:
    """
    Create and configure the Trainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs

    Returns:
        Trainer: Configured trainer
    """
    logger.info("Creating trainer...")

    # Setup training arguments
    training_args = setup_training_args(output_dir, num_epochs, has_validation=val_dataset is not None)

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Initialize trainer (MLflow autolog will handle callbacks automatically)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )

    logger.info("Trainer created successfully")
    if val_dataset:
        logger.info(f"Validation dataset configured with {len(val_dataset)} samples")

    return trainer


def save_model(trainer: Trainer, tokenizer: AutoTokenizer, output_dir: str):
    """
    Save the trained model and tokenizer.

    Args:
        trainer: The trainer with the trained model
        tokenizer: The tokenizer to save
        output_dir: Directory to save the model
    """
    logger.info(f"Saving model to {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the LoRA adapter and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log what was saved
    saved_files = list(Path(output_dir).glob("*"))
    logger.info(f"Saved files: {[f.name for f in saved_files]}")
    logger.info(f"Model saved successfully to {output_dir}")


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Get information about the model.

    Args:
        model: The model to analyze

    Returns:
        Dict with model information
    """
    try:
        # Get parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            "model_name": MODEL_NAME,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "device": str(model.device) if hasattr(model, "device") else "unknown",
        }

        logger.info("Model information:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable percentage: {info['trainable_percentage']:.2f}%")

        return info

    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return {"error": str(e)}
