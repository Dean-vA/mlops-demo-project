"""
Model setup and configuration for LoRA fine-tuning
Handles model loading, LoRA configuration, and training setup
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

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
    # Create full prompts
    prompts = [create_training_prompt(inp, target) for inp, target in zip(examples["input_text"], examples["target_summary"])]

    # Tokenize with truncation
    tokenized = tokenizer(prompts, truncation=True, padding=True, max_length=max_length, return_tensors=None)

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

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    logger.info(f"Created dataset with {len(dataset)} examples")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ["input_text", "target_summary"]],
    )

    # Log dataset statistics
    avg_length = np.mean([len(x) for x in tokenized_dataset["input_ids"]])
    max_length = max([len(x) for x in tokenized_dataset["input_ids"]])
    min_length = min([len(x) for x in tokenized_dataset["input_ids"]])

    logger.info("Dataset tokenization complete:")
    logger.info(f"  - Total examples: {len(tokenized_dataset)}")
    logger.info(f"  - Average sequence length: {avg_length:.0f}")
    logger.info(f"  - Max sequence length: {max_length}")
    logger.info(f"  - Min sequence length: {min_length}")

    return tokenized_dataset


def setup_training_args(output_dir: str, num_epochs: int = 10) -> TrainingArguments:
    """
    Configure training arguments.

    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs

    Returns:
        TrainingArguments: Configured training arguments
    """
    logger.info("Setting up training arguments...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,  # More epochs for small dataset
        per_device_train_batch_size=2,  # Small batch due to long sequences
        gradient_accumulation_steps=4,  # Effective batch size of 8
        learning_rate=2e-4,  # Standard LoRA learning rate
        weight_decay=0.01,
        logging_steps=1,  # Log every step for small dataset
        save_strategy="epoch",
        save_total_limit=3,
        warmup_steps=2,  # Small warmup for small dataset
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
        push_to_hub=False,  # Don't push to HuggingFace Hub
    )

    logger.info("Training arguments configured:")
    logger.info(f"  - Epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")

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
    )

    return data_collator


def create_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, output_dir: str, num_epochs: int = 10) -> Trainer:
    """
    Create and configure the Trainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Training dataset
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs

    Returns:
        Trainer: Configured trainer
    """
    logger.info("Creating trainer...")

    # Setup training arguments
    training_args = setup_training_args(output_dir, num_epochs)

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Trainer created successfully")
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
