"""
Simple unit tests for the transcription module.
"""

import pytest
from unittest.mock import Mock, patch

from backend_microservice.transcription import is_model_loaded

@pytest.fixture
def reset_model_state():
    """Fixture to reset the global model state before each test."""
    import backend_microservice.transcription as trans_module
    trans_module._model = None

@pytest.mark.usefixtures("reset_model_state")

def test_is_model_loaded_false():
    """Test is_model_loaded returns False when model is not loaded."""
    result = is_model_loaded()
    assert result is False


@pytest.mark.usefixtures("reset_model_state")
def test_is_model_loaded_true():
    """Test is_model_loaded returns True when model is loaded."""
    import backend_microservice.transcription as trans_module
    trans_module._model = Mock()
    
    result = is_model_loaded()
    assert result is True


@pytest.mark.usefixtures("reset_model_state")
@patch('backend_microservice.transcription.nemo_asr')
@patch('backend_microservice.transcription.torch')
def test_load_model_basic(mock_torch, mock_nemo_asr):
    """Test basic model loading functionality."""
    from backend_microservice.transcription import load_model
    import backend_microservice.transcription as trans_module
    
    # Reset model state
    trans_module._model = None
    
    # Mock dependencies
    mock_torch.cuda.is_available.return_value = True
    mock_model = Mock()
    # Make the .to() method return the same mock object
    mock_model.to.return_value = mock_model
    mock_nemo_asr.models.ASRModel.from_pretrained.return_value = mock_model
    
    # Load model
    result = load_model()
    
    # Assertions
    assert result == mock_model
    mock_nemo_asr.models.ASRModel.from_pretrained.assert_called_once_with(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    mock_model.to.assert_called_once_with("cuda")
    mock_model.eval.assert_called_once()
    
    # Clean up
    trans_module._model = None