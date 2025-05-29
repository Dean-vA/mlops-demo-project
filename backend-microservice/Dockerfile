# Use NVIDIA CUDA base image with cuDNN for Ubuntu 22.04
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set working directory
WORKDIR /app/src/backend_microservice

# Install Poetry
RUN pip install poetry==2.1.3

# Copy source code
COPY . /app

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --only main

# Install PyTorch with CUDA 12.8 support
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip install cuda-python>=12.3

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "backend_microservice.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ENTRYPOINT [ "/bin/bash" ]
