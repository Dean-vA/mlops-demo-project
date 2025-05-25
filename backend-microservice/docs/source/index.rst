.. MLOps Demo Project - Parakeet STT API documentation master file, created by
   sphinx-quickstart on Fri May 23 07:55:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Parakeet STT API Documentation
===============================

Welcome to the Parakeet STT API documentation!

This API provides speech-to-text transcription using NVIDIA's Parakeet TDT 0.6B v2 model.

Quick Start
-----------

1. Start the API:

   .. code-block:: bash

      cd backend-microservice
      poetry run uvicorn backend_microservice.main:app --reload

2. Make a request:

   .. code-block:: bash

      curl -X POST http://localhost:8000/transcribe \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_audio.wav"

API Endpoints
-------------

**GET /** - Welcome message

**GET /health** - Health check

**POST /transcribe** - Transcribe audio file

Parameters:
- ``file``: Audio file (.wav or .flac)
- ``return_timestamps``: Include timestamps (default: true)

Quick Start Development
-----------

Option 1: Docker (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   git clone <your-repo-url>
   cd mlops-demo-project/backend-microservice
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Check if the service is running**:
   ```bash
   curl http://localhost:3569/health
   ```

4. **Access the API documentation**:
   - Open your browser to http://localhost:3569/docs

Option 2: Local Development

1. **Navigate to the backend directory**:
   ```bash
   cd mlops-demo-project/backend-microservice
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```

   If you want to run on gpu you will need to install PyTorch with CUDA support. You can do this by running:

   ```bash
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Run the application**:
   ```bash
   poetry run uvicorn backend_microservice.main:app --host 0.0.0.0 --port 8000 --reload
   ```

Code Documentation
------------------

All modules are automatically documented below:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   backend_microservice
