"""
Simple Azure ML utilities using environment variables
"""

import logging
import os
from typing import Any, Dict, Optional

import dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential

# Load environment variables from .env file
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


def get_ml_client() -> Optional[MLClient]:
    """Get Azure ML client from environment variables."""
    try:
        credential = ClientSecretCredential(
            tenant_id=os.getenv("AZURE_TENANT_ID"), client_id=os.getenv("AZURE_CLIENT_ID"), client_secret=os.getenv("AZURE_CLIENT_SECRET")
        )

        return MLClient(
            credential=credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
        )
    except Exception as e:
        logger.error(f"Failed to create ML client: {e}")
        return None


def register_model(
    model_path: str, model_name: str, description: str, tags: Dict[str, str] = None, properties: Dict[str, Any] = None
) -> Optional[Dict[str, Any]]:
    """Register model in Azure ML."""
    ml_client = get_ml_client()
    if not ml_client:
        return None

    try:
        model = Model(name=model_name, path=model_path, description=description, type="custom_model", tags=tags or {}, properties=properties or {})

        registered_model = ml_client.models.create_or_update(model)

        return {
            "status": "success",
            "model_name": registered_model.name,
            "model_version": registered_model.version,
            "model_id": registered_model.id,
            "registration_time": registered_model.creation_context.created_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None
