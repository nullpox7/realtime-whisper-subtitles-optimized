# Real-time Whisper Subtitles - Stable Dockerfile (Fixed v2.3.0)
# Compatible with multiple CUDA versions and stable base images
# Author: Real-time Whisper Subtitles Team
# Encoding: UTF-8

FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LOG_PATH=/app/data/logs

# Numba optimization - stable settings
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/dev/null
ENV NUMBA_THREADING_LAYER=workqueue
ENV NUMBA_PARALLEL=0

# Locale settings
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    # System utilities
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch CPU version (stable)
RUN pip3 install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/{models,outputs,logs,cache} \
    && mkdir -p /app/static \
    && mkdir -p /app/templates \
    && mkdir -p /app/src \
    && mkdir -p /app/config \
    && mkdir -p /app/logs \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser \
    && chmod -R 755 /home/appuser

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# Copy environment examples
COPY .env.example .

# Set final permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create user-specific data directories
RUN mkdir -p /app/data/{models/whisper,outputs,logs,cache} \
    && mkdir -p /app/logs \
    && mkdir -p /home/appuser/.cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL=base
ENV LANGUAGE=auto
ENV DEVICE=auto
ENV COMPUTE_TYPE=auto
ENV BEAM_SIZE=1
ENV TEMPERATURE=0.0
ENV ENABLE_WORD_TIMESTAMPS=false
ENV VAD_FILTER=true
ENV BATCH_SIZE=4
ENV MAX_WORKERS=2

# Default command
CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]