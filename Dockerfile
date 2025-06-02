# Real-time Whisper Subtitles - Production Dockerfile
# CUDA 12.1 + cuDNN optimized for speech recognition
# Author: Real-time Whisper Subtitles Team
# Encoding: UTF-8

# Multi-stage build for optimized production image
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
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
    # Additional system tools
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Upgrade pip and install wheel
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# NVIDIA Runtime environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
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
    curl \
    wget \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy PyTorch installation from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch* /usr/local/lib/python${PYTHON_VERSION}/dist-packages/
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision* /usr/local/lib/python${PYTHON_VERSION}/dist-packages/
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchaudio* /usr/local/lib/python${PYTHON_VERSION}/dist-packages/

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
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
RUN chmod +x /app/src/*.py

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
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
