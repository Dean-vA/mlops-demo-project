"""
Simple unit tests for the FastAPI main application.
"""

from unittest.mock import patch

import asyncio
import pytest
from backend_microservice.main import app, health, root


@pytest.fixture
def event_loop():
    """Use a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_root_endpoint():
    """Test the root endpoint returns welcome message."""
    response = asyncio.get_event_loop().run_until_complete(root())
    assert response == {"message": "Welcome to the Parakeet STT API!"}


@patch("backend_microservice.main.is_model_loaded")
def test_health_endpoint_model_loaded(mock_is_loaded):
    """Test health endpoint when model is loaded."""
    mock_is_loaded.return_value = True

    response = asyncio.get_event_loop().run_until_complete(health())

    assert response["status"] == "healthy"
    assert response["model_loaded"] is True


@patch("backend_microservice.main.is_model_loaded")
def test_health_endpoint_model_not_loaded(mock_is_loaded):
    """Test health endpoint when model is not loaded."""
    mock_is_loaded.return_value = False

    response = asyncio.get_event_loop().run_until_complete(health())

    assert response["status"] == "healthy"
    assert response["model_loaded"] is False


def test_nonexistent_endpoint():
    """Test that a non-existent endpoint is not registered."""
    paths = [route.path for route in app.router.routes]
    assert "/nonexistent" not in paths
