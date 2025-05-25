# MLOps Demo Project - Parakeet STT API

A FastAPI microservice for speech-to-text transcription using NVIDIA's Parakeet TDT 0.6B v2 model. This project demonstrates a production-ready MLOps implementation with containerization and modern Python practices.

## 🚀 Features

- **High-Quality Transcription**: Uses NVIDIA's state-of-the-art Parakeet TDT 0.6B v2 model
- **Word-Level Timestamps**: Provides precise timing information for each word
- **Multiple Audio Formats**: Supports `.wav` and `.flac` audio files
- **Docker Support**: Fully containerized for easy deployment
- **Production Ready**: Includes health checks, logging, and error handling
- **Modern Python**: Built with FastAPI, Poetry, and Python 3.11+

## 📁 Project Structure

```
mlops-demo-project/
├── backend-microservice/
│   ├── docs
│   ├── src/
│   │   └── backend_microservice/
│   │       ├── __init__.py
│   │       ├── main.py              # FastAPI application
│   │       └── transcription.py     # STT model integration
│   ├── tests/
│   │   └── __init__.py
│   │   ├── test_main.py        # Unit tests for main app
│   │   └── test_transcription.py # Unit tests for transcription
|   ├── .pre-commit-config.yaml # Pre-commit hooks configuration
│   ├── docker-compose.yml           # Docker Compose configuration
│   ├── Dockerfile                   # Container definition
│   ├── pyproject.toml              # Poetry dependencies
│   └── poetry.lock                 # Locked dependencies
├── .gitignore
└── README.md
```

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **Language**: Python 3.11+
- **ML Model**: NVIDIA Parakeet TDT 0.6B v2
- **Package Manager**: Poetry
- **Containerization**: Docker & Docker Compose
- **ML Framework**: PyTorch + NeMo Toolkit

## 📋 Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for optimal performance)

## 🚀 Quick Start

### Option 1: Docker (Recommended)

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

### Option 2: Local Development

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

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Root Endpoint
```http
GET /
```

**Response**:
```json
{
  "message": "Welcome to the Parakeet STT API!"
}
```

### Transcribe Audio
```http
POST /transcribe
```

**Parameters**:
- `file` (required): Audio file (`.wav` or `.flac`)
- `return_timestamps` (optional): Include word-level timestamps (default: `true`)
- `chunk_duration_sec` (optional): Duration for processing chunks
- `overlap_duration_sec` (optional): Overlap duration between chunks

**Example Response**:
```json
{
  "text": "Hello world, this is a test transcription.",
  "processing_time_sec": 1.23,
  "timestamps": {
    "word": [
      {"word": "Hello", "start": 0.0, "end": 0.5},
      {"word": "world", "start": 0.5, "end": 0.9}
    ],
    "segment": [...]
  },
  "segments": [
    {
      "start": 0.0,
      "end": 2.1,
      "text": "Hello world, this is a test transcription."
    }
  ]
}
```

## 📝 Usage Examples

### Using cURL

**Basic transcription**:
```bash
curl -X POST http://localhost:3569/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
```

**With timestamps disabled**:
```bash
curl -X POST http://localhost:3569/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav" \
  -F "return_timestamps=false"
```

### Using Python

```python
import requests

# Transcribe an audio file
with open("your_audio.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:3569/transcribe",
        files={"file": audio_file},
        data={"return_timestamps": True}
    )

result = response.json(
