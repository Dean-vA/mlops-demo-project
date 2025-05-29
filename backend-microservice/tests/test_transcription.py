"""
Simple unit tests for the transcription module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
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
@patch("backend_microservice.transcription.get_device")
@patch("backend_microservice.transcription.nemo_asr")
@patch("backend_microservice.transcription.torch")
def test_load_model_basic(mock_torch, mock_nemo_asr, mock_get_device):
    """Test basic model loading functionality."""
    import backend_microservice.transcription as trans_module
    from backend_microservice.transcription import load_model

    # Reset model state
    trans_module._model = None

    # Mock get_device to return 'cuda'
    mock_get_device.return_value = "cuda"

    # Mock torch CUDA availability
    mock_torch.cuda.is_available.return_value = True

    # Create a more sophisticated mock model
    mock_model = MagicMock()

    # Mock the .to() method to return the same mock object
    mock_model.to.return_value = mock_model

    # Mock the .half() method for FP16 conversion
    mock_model.half.return_value = mock_model

    # Mock the .parameters() method to return an iterable of mock parameters
    # This fixes the TypeError: 'Mock' object is not iterable
    mock_param1 = Mock()
    mock_param1.numel.return_value = 1000
    mock_param2 = Mock()
    mock_param2.numel.return_value = 2000
    mock_model.parameters.return_value = [mock_param1, mock_param2]

    # Set up nemo_asr mock
    mock_nemo_asr.models.ASRModel.from_pretrained.return_value = mock_model

    # Load model
    result = load_model()

    # Assertions
    assert result == mock_model
    mock_get_device.assert_called_once()
    mock_nemo_asr.models.ASRModel.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v2")
    mock_model.to.assert_called_once_with("cuda")
    mock_model.eval.assert_called_once()

    # GPU optimization assertions
    mock_model.half.assert_called_once()  # FP16 conversion
    assert mock_torch.backends.cudnn.benchmark is True  # cuDNN autotuner

    # Clean up
    trans_module._model = None


@pytest.mark.usefixtures("reset_model_state")
@patch("backend_microservice.transcription.get_device")
@patch("backend_microservice.transcription.nemo_asr")
@patch("backend_microservice.transcription.torch")
def test_load_model_cpu(mock_torch, mock_nemo_asr, mock_get_device):
    """Test model loading on CPU (no GPU optimizations)."""
    import backend_microservice.transcription as trans_module
    from backend_microservice.transcription import load_model

    # Reset model state
    trans_module._model = None

    # Mock get_device to return 'cpu'
    mock_get_device.return_value = "cpu"

    # Mock torch CUDA not available
    mock_torch.cuda.is_available.return_value = False

    # Create a mock model
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    # Mock parameters for counting
    mock_param = Mock()
    mock_param.numel.return_value = 3000
    mock_model.parameters.return_value = [mock_param]

    # Set up nemo_asr mock
    mock_nemo_asr.models.ASRModel.from_pretrained.return_value = mock_model

    # Load model
    result = load_model()

    # Assertions
    assert result == mock_model
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()

    # Ensure GPU optimizations are NOT called on CPU
    mock_model.half.assert_not_called()

    # Clean up
    trans_module._model = None
