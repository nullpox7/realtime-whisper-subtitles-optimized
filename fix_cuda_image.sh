#!/bin/bash
# CUDA Image Fix Script for Real-time Whisper Subtitles
# Automatically detects and fixes CUDA Docker image issues
# Encoding: UTF-8

set -e

echo "? CUDA Image Fix for Real-time Whisper Subtitles"
echo "================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker image exists
check_docker_image() {
    local image="$1"
    log_info "Checking: $image"
    
    if docker manifest inspect "$image" >/dev/null 2>&1; then
        log_success "? Available: $image"
        return 0
    else
        log_warning "? Not found: $image"
        return 1
    fi
}

# Check if we're in the right directory
if [ ! -f "docker-compose.gpu.yml" ] && [ ! -f "Dockerfile.gpu" ]; then
    log_error "This script must be run from the project root directory"
    exit 1
fi

log_info "Starting CUDA image compatibility check..."
echo ""

# Array of CUDA images to test in order of preference
CUDA_IMAGES=(
    "nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04"
    "nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04" 
    "nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
)

# Check which images are available
available_image=""
for image in "${CUDA_IMAGES[@]}"; do
    if check_docker_image "$image"; then
        available_image="$image"
        break
    fi
done

echo ""

if [ -z "$available_image" ]; then
    log_error "No compatible CUDA images found!"
    echo ""
    log_info "Troubleshooting steps:"
    echo "1. Check internet connection"
    echo "2. Try: docker login"
    echo "3. Update Docker: docker version"
    echo "4. Manual pull: docker pull nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04"
    exit 1
fi

log_success "Found compatible image: $available_image"
echo ""

# Determine CUDA version for optimization
case "$available_image" in
    *12.9*)
        cuda_version="12.9"
        pytorch_version="torch==2.4.1+cu124"
        ;;
    *12.4*)
        cuda_version="12.4"
        pytorch_version="torch==2.4.1+cu124"
        ;;
    *12.2*)
        cuda_version="12.2"
        pytorch_version="torch==2.3.1+cu121"
        ;;
    *12.1*)
        cuda_version="12.1"
        pytorch_version="torch==2.3.1+cu121"
        ;;
    *)
        cuda_version="12.4"
        pytorch_version="torch==2.4.1+cu124"
        ;;
esac

log_info "Detected CUDA version: $cuda_version"
log_info "Compatible PyTorch: $pytorch_version"
echo ""

# Create optimized Dockerfile.gpu.fixed
log_info "Creating optimized Dockerfile.gpu.fixed..."

cat > Dockerfile.gpu.fixed << EOF
# Real-time Whisper Subtitles - GPU Optimized (Auto-Fixed)
# Using compatible CUDA image: $available_image
# Author: Real-time Whisper Subtitles Team
# Encoding: UTF-8

FROM $available_image

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LOG_PATH=/app/data/logs

# CUDA optimization environment variables
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Numba optimization for GPU
ENV NUMBA_DISABLE_JIT=0
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_THREADING_LAYER=workqueue
ENV NUMBA_PARALLEL=1

# CUDA specific optimizations
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential cmake git curl wget pkg-config \\
    python3.11 python3.11-dev python3-pip \\
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \\
    libswscale-dev libswresample-dev libasound2-dev \\
    portaudio19-dev libportaudio2 libportaudiocpp0 libpulse-dev \\
    htop nvtop libnccl2 libnccl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \\
    ln -s /usr/bin/python3.11 /usr/bin/python

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.gpu.txt .

# Upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with compatible CUDA version
RUN python3 -m pip install $pytorch_version torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.gpu.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/{models,outputs,logs,cache} /app/static /app/templates /app/src /app/config /app/logs \\
    && mkdir -p /tmp/cuda_cache /tmp/numba_cache /home/appuser/.cache /home/appuser/.torch \\
    && chown -R appuser:appuser /app /home/appuser /tmp/cuda_cache /tmp/numba_cache \\
    && chmod -R 755 /tmp/cuda_cache /tmp/numba_cache /home/appuser

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY .env.gpu.example* ./

# Set final permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create user-specific data directories
RUN mkdir -p /app/data/{models/whisper,outputs,logs,cache} /app/logs \\
    && mkdir -p /home/appuser/.cache/torch /home/appuser/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=180s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables optimized for GPU and large-v3
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL=large-v3
ENV LANGUAGE=auto
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV BEAM_SIZE=5
ENV TEMPERATURE=0.0
ENV BEST_OF=5
ENV ENABLE_WORD_TIMESTAMPS=true
ENV VAD_FILTER=true
ENV BATCH_SIZE=16
ENV MAX_WORKERS=4

# CUDA memory optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_MEMORY_FRACTION=0.85

# Default command
CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

log_success "Created: Dockerfile.gpu.fixed"
echo ""

# Update docker-compose.gpu.yml if needed
if [ -f "docker-compose.gpu.yml" ]; then
    log_info "Updating docker-compose.gpu.yml to use fixed Dockerfile..."
    
    # Create backup
    cp docker-compose.gpu.yml docker-compose.gpu.yml.backup
    
    # Update the dockerfile reference
    sed -i 's/dockerfile: Dockerfile.gpu$/dockerfile: Dockerfile.gpu.fixed/' docker-compose.gpu.yml
    
    log_success "Updated docker-compose.gpu.yml"
    log_info "Backup saved as: docker-compose.gpu.yml.backup"
    echo ""
fi

# Provide next steps
echo "? Next Steps:"
echo "=============="
echo ""
echo "1. Build the fixed GPU image:"
echo "   docker-compose -f docker-compose.gpu.yml build --no-cache whisper-subtitles-gpu"
echo ""
echo "2. Start the application:"
echo "   docker-compose -f docker-compose.gpu.yml up -d"
echo ""
echo "3. Verify GPU access:"
echo "   docker-compose -f docker-compose.gpu.yml exec whisper-subtitles-gpu nvidia-smi"
echo ""
echo "4. Check application health:"
echo "   curl http://localhost:8000/health"
echo ""

# Optional: Auto-build
read -p "Do you want to build the fixed image now? (y/N): " -n 1 -r
echo
if [[ \$REPLY =~ ^[Yy]\$ ]]; then
    log_info "Building fixed GPU image..."
    
    if docker-compose -f docker-compose.gpu.yml build --no-cache whisper-subtitles-gpu; then
        log_success "GPU image built successfully!"
        echo ""
        
        read -p "Start the application now? (y/N): " -n 1 -r
        echo
        if [[ \$REPLY =~ ^[Yy]\$ ]]; then
            log_info "Starting application..."
            docker-compose -f docker-compose.gpu.yml up -d
            
            log_info "Waiting for startup (60 seconds)..."
            sleep 60
            
            if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                log_success "? Application is running with GPU acceleration!"
                echo ""
                echo "Access your application at: http://localhost:8000"
                echo "GPU monitoring: docker-compose -f docker-compose.gpu.yml exec whisper-subtitles-gpu nvidia-smi"
            else
                log_warning "Application started but health check failed"
                echo "Check logs: docker-compose -f docker-compose.gpu.yml logs whisper-subtitles-gpu"
            fi
        fi
    else
        log_error "Build failed. Check the error messages above."
        exit 1
    fi
fi

echo ""
log_success "CUDA image fix completed!"
echo ""
echo "? Summary:"
echo "- Compatible CUDA image: $available_image"
echo "- CUDA version: $cuda_version"
echo "- PyTorch version: $pytorch_version"
echo "- Fixed Dockerfile: Dockerfile.gpu.fixed"
echo ""
echo "For further assistance, check the project documentation or GitHub issues."