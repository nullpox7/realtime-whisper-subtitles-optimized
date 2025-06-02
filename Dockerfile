# Real-time Whisper Subtitles - Production Dockerfile
# CUDA optimized for speech recognition
# Author: Real-time Whisper Subtitles Team
# Encoding: UTF-8

# Use Ubuntu base with CUDA runtime
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# NVIDIA Runtime environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    # Audio processing dependencies
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libpulse-dev \
    # Additional tools
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support (11.8 compatible)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/{models,outputs,logs,cache} \
    && mkdir -p /app/static \
    && mkdir -p /app/templates \
    && mkdir -p /app/src \
    && mkdir -p /app/config

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY .env.example ./

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create data directories with proper permissions
RUN mkdir -p /app/data/{models/whisper,outputs,logs,cache}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL=base
ENV LANGUAGE=ja
ENV DEVICE=auto

# Default command
CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
