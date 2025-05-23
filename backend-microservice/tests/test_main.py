"""
Simple unit tests for the FastAPI main application.
"""

from unittest.mock import patch

import pytest
from backend_microservice.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Parakeet STT API!"}


@patch("backend_microservice.main.is_model_loaded")
def test_health_endpoint_model_loaded(mock_is_loaded, client):
    """Test health endpoint when model is loaded."""
    mock_is_loaded.return_value = True

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@patch("backend_microservice.main.is_model_loaded")
def test_health_endpoint_model_not_loaded(mock_is_loaded, client):
    """Test health endpoint when model is not loaded."""
    mock_is_loaded.return_value = False

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


def test_nonexistent_endpoint(client):
    """Test 404 response for non-existent endpoints."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
