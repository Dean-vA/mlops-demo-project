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

Code Documentation
------------------

All modules are automatically documented below:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   backend_microservice

