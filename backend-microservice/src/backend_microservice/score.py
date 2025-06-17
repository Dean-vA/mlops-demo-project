"""
Azure ML Endpoint Scoring Script for D&D LoRA Summarization Model
"""

import json
import logging
import os

from dotenv import load_dotenv

# Import summarization functions
from summarise import generate_summary, get_model_info, load_model

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_path = None
model_loaded = False


def init():
    """
    Initialize the model when the endpoint starts.
    This function is called once when the endpoint is deployed.
    """
    global model_path, model_loaded

    logger.info("Initializing D&D LoRA summarization endpoint...")

    try:
        # Set up Hugging Face authentication if token is provided
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            logger.info("Setting up Hugging Face authentication...")
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            # Also try the login approach
            try:
                from huggingface_hub import login

                login(token=hf_token)
                logger.info("Successfully logged into Hugging Face")
            except Exception as e:
                logger.warning(f"Could not login to HF Hub: {e}")
        else:
            logger.warning("No HF_TOKEN found in environment variables")

        # Get model path from environment variable
        model_path = os.environ.get("AZUREML_MODEL_DIR")
        print(f"base_path: {model_path}")

        # Check if model directory exists and find the correct path
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            # Try alternative paths
            alternative_paths = [
                "./model",
                "./downloaded_model/lora-session-summary-model",
                "./lora-session-summary-model",
                "/var/azureml-app/azureml-models/dnd-lora-model/1",
                "/var/azureml-app/azureml-models//a70e9ebae6cc225e645c1489357f0ac9/1/lora-session-summary-model/INPUT_model",
            ]

            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    logger.info(f"Found model at alternative path: {model_path}")
                    break
            else:
                raise FileNotFoundError("Model not found in any expected location")

        # Look for the actual LoRA adapter files
        possible_model_paths = [
            model_path,
            os.path.join(model_path, "INPUT_model"),  # This is the missing path!
            os.path.join(model_path, "INPUT_model", "checkpoint-20"),  # Latest checkpoint
            os.path.join(model_path, "INPUT_model", "checkpoint-18"),  # Alternative checkpoint
            os.path.join(model_path, "lora-session-summary-model"),
            os.path.join(model_path, "lora-session-summary-model", "INPUT_model"),
            os.path.join(model_path, "1", "lora-session-summary-model", "INPUT_model"),
        ]

        actual_model_path = None
        for path in possible_model_paths:
            adapter_config_path = os.path.join(path, "adapter_config.json")
            logger.info(f"Checking for adapter_config.json at: {adapter_config_path}")
            if os.path.exists(adapter_config_path):
                actual_model_path = path
                logger.info(f"Found LoRA adapter at: {actual_model_path}")
                break

        if not actual_model_path:
            # List all files in model_path for debugging
            logger.error("Could not find adapter_config.json. Listing model directory contents:")
            for root, dirs, files in os.walk(model_path):
                level = root.replace(model_path, "").count(os.sep)
                indent = " " * 2 * level
                logger.error(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    logger.error(f"{subindent}{file}")
            raise FileNotFoundError("Could not find adapter_config.json in any expected location")

        model_path = actual_model_path
        logger.info(f"Using model path: {model_path}")

        # Pre-load the model for faster inference
        logger.info("Pre-loading model...")
        load_model(model_path)
        model_loaded = True

        # Get model info for logging
        info = get_model_info()
        logger.info(f"Model initialization complete: {info}")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model_loaded = False
        raise


def run(raw_data: str) -> str:
    """
    Process inference requests.

    Args:
        raw_data: JSON string containing the input data

    Returns:
        JSON string containing the model predictions
    """
    logger.info("Processing inference request...")

    try:
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Received data keys: {list(data.keys())}")

        # print json data for debugging
        logger.debug(f"Input data: {json.dumps(data, indent=2)}")

        # Check if model is loaded
        if not model_loaded:
            logger.error("Model not properly initialized")
            return json.dumps({"error": "Model not initialized"}), 500

        # Extract transcript from input data
        if "transcript" in data:
            transcript = data["transcript"]
        elif "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            transcript = data["data"][0]
        else:
            logger.error("No transcript found in input data")
            return json.dumps({"error": "No transcript provided"}), 400

        # Generate summary
        logger.info("Generating summary...")
        summary = generate_summary(transcript, model_path)

        # Return response
        response = {"summary": summary, "status": "success", "model_info": get_model_info()}

        logger.info("Summary generation completed successfully")
        return json.dumps(response)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": "Invalid JSON input"}), 400
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return json.dumps({"error": f"Inference failed: {str(e)}"}), 500
