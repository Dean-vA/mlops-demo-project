# MLOps Demo Project - Parakeet STT API

A FastAPI microservice for speech-to-text transcription using NVIDIA's Parakeet TDT 0.6B v2 model with real-time speaker diarization capabilities. This project demonstrates a production-ready MLOps implementation with containerization, automated testing, and comprehensive documentation.

## 🚀 Features

- **High-Quality Transcription**: Uses NVIDIA's state-of-the-art Parakeet TDT 0.6B v2 model
- **Real Speaker Diarization**: Identifies and separates different speakers using NeMo's ClusteringDiarizer
- **Word-Level Timestamps**: Provides precise timing information for each word
- **Multiple Audio Formats**: Supports `.wav` and `.flac` audio files
- **Docker Support**: Fully containerized for easy deployment
- **Production Ready**: Includes health checks, logging, error handling, and GPU optimization
- **Modern Python**: Built with FastAPI, Poetry, and Python 3.11+
- **Interactive Frontend**: Web interface for easy audio upload and transcription
- **Comprehensive Testing**: Unit tests with coverage reporting
- **Auto-Generated Documentation**: Complete API documentation with Sphinx

## 📖 Documentation

- **Live API Documentation**: [https://dean-va.github.io/mlops-demo-project/](https://dean-va.github.io/mlops-demo-project/)
- **Interactive API Docs**: Available at `/docs` when running locally (e.g., http://localhost:3569/docs)
- **Frontend Interface**: Available at port 3570 when running with Docker Compose

## 📁 Project Structure

```
mlops-demo-project/
├── backend-microservice/           # Main API service
│   ├── docs/                      # Sphinx documentation
│   ├── src/backend_microservice/  # Source code
│   │   ├── main.py               # FastAPI application
│   │   ├── transcription.py      # STT model integration
│   │   ├── diarization.py        # Speaker diarization
│   │   ├── gpu_utils.py          # GPU utilities
│   │   └── audio_utils.py        # Audio processing
│   ├── tests/                    # Unit tests
│   ├── .pre-commit-config.yaml   # Code quality hooks
│   ├── docker-compose.yml        # Docker services
│   ├── Dockerfile               # Container definition
│   ├── pyproject.toml           # Poetry dependencies
│   └── poetry.lock              # Locked dependencies
├── frontend/                     # Web interface
│   ├── index.html               # Interactive frontend
│   ├── nginx.conf               # Web server config
│   └── Dockerfile               # Frontend container
├── data/                        # Data processing scripts
│   ├── generate_transcripts.py  # Batch transcription
│   ├── combine_transcripts.py   # Transcript processing
│   └── generate_summaries_gpt.py # AI summarization
├── .github/workflows/           # CI/CD pipelines
│   ├── tests.yml               # Automated testing
│   └── docs.yml                # Documentation deployment
├── train.csv                   # Training data sample
└── README.md                   # This file
```

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **Language**: Python 3.11+
- **ML Models**:
  - NVIDIA Parakeet TDT 0.6B v2 (Speech-to-Text)
  - NeMo ClusteringDiarizer with TitaNet + MarbleNet VAD (Speaker Diarization)
- **Package Manager**: Poetry
- **Containerization**: Docker & Docker Compose
- **ML Framework**: PyTorch + NeMo Toolkit
- **Frontend**: Vanilla HTML/CSS/JavaScript with modern UI
- **Documentation**: Sphinx with auto-generated API docs
- **Testing**: pytest with coverage
- **CI/CD**: GitHub Actions

## 📋 Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for optimal performance)

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   git clone https://github.com/Dean-vA/mlops-demo-project.git
   cd mlops-demo-project
   ```

2. **Start all services with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the services**:
   - **API**: http://localhost:3569
   - **API Documentation**: http://localhost:3569/docs
   - **Web Interface**: http://localhost:3570
   - **Health Check**: http://localhost:3569/health

### Option 2: Local Development

1. **Navigate to the backend directory**:
   ```bash
   cd mlops-demo-project/backend-microservice
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```

3. **For GPU support, install CUDA-enabled PyTorch**:
   ```bash
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Run the application**:
   ```bash
   poetry run uvicorn backend_microservice.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open the frontend** (optional):
   ```bash
   cd ../frontend
   # Serve the HTML file using your preferred method
   python -m http.server 8080
   ```

## 🔌 API Endpoints

### Core Endpoints

- **`GET /`** - Welcome message
- **`GET /health`** - Health check with model and GPU status
- **`GET /gpu-info`** - Detailed GPU information

### Transcription Endpoints

- **`POST /transcribe`** - Speech-to-text transcription only
- **`POST /diarize`** - Speaker diarization only
- **`POST /transcribe_and_diarize`** - Combined transcription + speaker identification

### Parameters

- `file` (required): Audio file (`.wav` or `.flac`)
- `return_timestamps` (optional): Include word-level timestamps (default: `true`)
- `num_speakers` (optional): Number of speakers for diarization (auto-detect if not specified)
- `chunk_duration_sec` (optional): Duration for processing chunks
- `overlap_duration_sec` (optional): Overlap duration between chunks

## 📝 Usage Examples

### Using cURL

**Basic transcription**:
```bash
curl -X POST http://localhost:3569/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
```

**Transcription with speaker diarization**:
```bash
curl -X POST http://localhost:3569/transcribe_and_diarize \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav" \
  -F "num_speakers=2"
```

### Using Python

```python
import requests

# Transcription with speaker diarization
with open("your_audio.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:3569/transcribe_and_diarize",
        files={"file": audio_file},
        data={"return_timestamps": True, "num_speakers": 2}
    )

result = response.json()
print("Transcription:", result["transcription"]["text"])
print("Speakers detected:", result["diarization"]["num_speakers_detected"])
```

### Example Response

```json
{
  "transcription": {
    "text": "Hello everyone, welcome to today's meeting.",
    "processing_time_sec": 2.34,
    "segments": [
      {
        "start": 0.0,
        "end": 3.2,
        "text": "Hello everyone, welcome to today's meeting.",
        "speaker": "Speaker 1"
      }
    ]
  },
  "diarization": {
    "num_speakers_detected": 2,
    "speakers": {
      "speaker_0": [{"start": 0.0, "end": 3.2, "duration": 3.2}],
      "speaker_1": [{"start": 3.5, "end": 8.1, "duration": 4.6}]
    }
  }
}
```

## 🧪 Testing

Run the test suite:

```bash
cd backend-microservice

# Run tests with coverage
poetry run pytest --cov=backend_microservice --cov-report=html --cov-report=term

# Run tests only
poetry run pytest

# Run specific test file
poetry run pytest tests/test_main.py
```

View coverage report: `htmlcov/index.html`

## 📚 Documentation

### Generate Documentation Locally

```bash
cd backend-microservice/docs

# Clean and rebuild docs
rm -rf build/ source/_autosummary/
poetry run sphinx-build -b html source build/html

# Open documentation
open build/html/index.html  # macOS
# or
start build/html/index.html  # Windows
```

### Documentation Features

- **Auto-generated API reference** from docstrings
- **Interactive examples** with code samples
- **Module documentation** for all components
- **Deployment guides** for different environments
- **Architecture overview** and design decisions

## 🐳 Docker Configuration

### Backend Service
- **Port**: 3569
- **GPU Support**: NVIDIA CUDA runtime
- **Health Checks**: Automated container health monitoring
- **Shared Memory**: 2GB for NeMo diarization models

### Frontend Service
- **Port**: 3570
- **Web Server**: Nginx with optimized configuration
- **Features**: File upload, real-time transcription, speaker visualization

## 🔧 Development

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run hooks manually
poetry run pre-commit run --all-files
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Write tests**: Add tests in `tests/` directory
3. **Update documentation**: Add docstrings and update docs if needed
4. **Run quality checks**: `poetry run pre-commit run --all-files`
5. **Submit PR**: Create pull request with description

### Data Processing Scripts

The `data/` directory contains utilities for processing audio data:

- **`generate_transcripts.py`**: Batch transcription of audio files
- **`combine_transcripts.py`**: Combine and process transcript segments
- **`generate_summaries_gpt.py`**: Generate AI summaries using OpenAI GPT

## 🚀 Deployment

### Production Deployment

1. **Set environment variables**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export NVIDIA_VISIBLE_DEVICES=all
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Monitor health**:
   ```bash
   curl http://your-server:3569/health
   ```

### Scaling Considerations

- **GPU Memory**: Model requires ~2GB VRAM
- **Processing Time**: ~1-2 seconds per minute of audio
- **Concurrent Requests**: Limited by GPU memory
- **Storage**: Temporary files cleaned automatically

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NVIDIA** for the Parakeet TDT model and NeMo toolkit
- **FastAPI** team for the excellent web framework
- **Poetry** for elegant dependency management
- **Sphinx** for documentation generation

## 📞 Support

- **Documentation**: [https://dean-va.github.io/mlops-demo-project/](https://dean-va.github.io/mlops-demo-project/)
- **Issues**: [GitHub Issues](https://github.com/Dean-vA/mlops-demo-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Dean-vA/mlops-demo-project/discussions)

---

⭐ **Star this repository if you find it helpful!**
