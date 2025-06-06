#!/bin/bash
# Real-time Whisper Subtitles - Memory Corruption Fix Script
# Fixes "corrupted double-linked list" error with memory-safe configuration
# Encoding: UTF-8

set -e

echo "Real-time Whisper Subtitles - Memory Corruption Fix"
echo "=================================================="
echo ""
echo "This script fixes the 'corrupted double-linked list' error by:"
echo "  - Implementing memory-safe Docker configuration"
echo "  - Disabling problematic JIT compilation"
echo "  - Using stable library versions"
echo "  - Applying thread-safe settings"
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

# Check if running in project directory
if [ ! -f "docker-compose.yml" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Stop all running containers
log_info "Stopping all running containers..."
docker-compose down --remove-orphans 2>/dev/null || true
docker-compose -f docker-compose.gpu.yml down --remove-orphans 2>/dev/null || true
log_success "Containers stopped"

# Step 2: Backup existing configuration
log_info "Backing up existing configuration..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

files_to_backup=(
    "Dockerfile.gpu"
    "docker-compose.gpu.yml" 
    "requirements.gpu.txt"
    "src/web_interface.py"
    ".env"
)

for file in "${files_to_backup[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$backup_dir/"
        log_info "Backed up $file"
    fi
done

log_success "Configuration backed up to $backup_dir/"

# Step 3: Create memory-safe environment configuration
log_info "Creating memory-safe environment configuration..."
cat > .env.memory_safe << 'EOF'
# Memory-Safe Configuration for Real-time Whisper Subtitles
# Eliminates memory corruption issues

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Conservative Model Configuration
WHISPER_MODEL=base
LANGUAGE=auto
DEVICE=cuda
COMPUTE_TYPE=float16

# Memory-Safe Processing Settings
BEAM_SIZE=1
BEST_OF=1
TEMPERATURE=0.0
BATCH_SIZE=1
MAX_WORKERS=1

# Disable Problematic Features
ENABLE_WORD_TIMESTAMPS=false
VAD_FILTER=false
ENABLE_QUANTIZATION=false

# Memory Safety Environment Variables
NUMBA_DISABLE_JIT=1
NUMBA_CACHE_DIR=/dev/null
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1

# CUDA Memory Safety
CUDA_MEMORY_FRACTION=0.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Thread Safety
WORKERS=1
THREADS=1

# Paths
MODEL_PATH=/app/data/models
OUTPUT_PATH=/app/data/outputs
LOG_PATH=/app/data/logs
LOG_LEVEL=INFO

# Security
RATE_LIMIT=50
MAX_UPLOAD_SIZE=50
ENABLE_CORS=true
CORS_ORIGINS=*

# Localization
TZ=UTC
LANG=C.UTF-8
LC_ALL=C.UTF-8
EOF

# Copy to main .env if user agrees
echo ""
read -p "Replace current .env with memory-safe configuration? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp .env.memory_safe .env
    log_success "Environment configuration updated"
else
    log_info "Memory-safe configuration saved as .env.memory_safe"
fi

# Step 4: Create memory-safe Dockerfile.gpu
log_info "Creating memory-safe GPU Dockerfile..."
cat > Dockerfile.gpu.safe << 'EOF'
# Real-time Whisper Subtitles - Memory-Safe GPU Dockerfile
# Eliminates "corrupted double-linked list" errors
# Author: Real-time Whisper Subtitles Team
# Encoding: UTF-8

FROM nvidia/cuda:12.1-base-ubuntu22.04

# Memory-safe environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# CRITICAL: Complete JIT disabling to prevent memory corruption
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/dev/null
ENV NUMBA_THREADING_LAYER=synchronous
ENV NUMBA_NUM_THREADS=1

# Memory allocator safety
ENV MALLOC_ARENA_MAX=2
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_MMAP_THRESHOLD_=131072

# Thread safety
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV BLIS_NUM_THREADS=1

# CUDA conservative settings
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=0
ENV CUDA_MEMORY_FRACTION=0.5
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Locale settings
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update and install essential packages only
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    build-essential \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.gpu.safe.txt requirements.txt

# Install Python packages with conservative versions
RUN python3.11 -m pip install --upgrade pip==24.0
RUN python3.11 -m pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN python3.11 -m pip install -r requirements.txt

# Create directories
RUN mkdir -p /app/data/{models,outputs,logs,cache} \
    && mkdir -p /app/src /app/static /app/templates \
    && chown -R appuser:appuser /app

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Conservative defaults
ENV WHISPER_MODEL=base
ENV BATCH_SIZE=1
ENV MAX_WORKERS=1
ENV DEVICE=cuda

# Start command
CMD ["python3.11", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

log_success "Memory-safe Dockerfile created: Dockerfile.gpu.safe"

# Step 5: Create memory-safe requirements
log_info "Creating memory-safe requirements file..."
cat > requirements.gpu.safe.txt << 'EOF'
# Memory-Safe GPU Requirements for Real-time Whisper Subtitles
# Conservative versions to prevent memory corruption

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==11.0.3
jinja2==3.1.2
python-multipart==0.0.6

# PyTorch (installed separately)
# torch==2.1.0+cu121
# torchaudio==2.1.0+cu121

# Whisper (conservative version)
faster-whisper==0.9.0

# Audio Processing (minimal, stable versions)
librosa==0.10.1
soundfile==0.12.1

# Scientific Computing (stable versions)
numpy==1.24.4
scipy==1.11.4

# System utilities
psutil==5.9.6
python-dotenv==1.0.0
requests==2.31.0

# Essential utilities only
tqdm==4.66.1
packaging==23.2

# Redis (optional)
redis==5.0.1

# Configuration
pyyaml==6.0.1

# Date/time
python-dateutil==2.8.2

# Monitoring (minimal)
prometheus-client==0.19.0
EOF

log_success "Memory-safe requirements created: requirements.gpu.safe.txt"

# Step 6: Create memory-safe docker-compose configuration
log_info "Creating memory-safe Docker Compose configuration..."
cat > docker-compose.gpu.safe.yml << 'EOF'
# Memory-Safe GPU Docker Compose Configuration
# Eliminates memory corruption issues

services:
  whisper-subtitles-safe:
    build:
      context: .
      dockerfile: Dockerfile.gpu.safe
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data/models:/app/data/models
      - ./data/outputs:/app/data/outputs
      - ./data/logs:/app/data/logs
    environment:
      # Memory safety critical settings
      - NUMBA_DISABLE_JIT=1
      - NUMBA_CACHE_DIR=/dev/null
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      
      # Conservative processing
      - WHISPER_MODEL=base
      - BATCH_SIZE=1
      - MAX_WORKERS=1
      - BEAM_SIZE=1
      - DEVICE=cuda
      - COMPUTE_TYPE=float16
      
      # Memory limits
      - CUDA_MEMORY_FRACTION=0.5
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
      
      # Application settings
      - HOST=0.0.0.0
      - PORT=8000
      - DEBUG=false
      - LOG_LEVEL=INFO
      
      # Paths
      - MODEL_PATH=/app/data/models
      - OUTPUT_PATH=/app/data/outputs
      - LOG_PATH=/app/data/logs
    
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    networks:
      - whisper-network
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7.0-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    networks:
      - whisper-network

networks:
  whisper-network:
    driver: bridge

volumes:
  redis_data:
EOF

log_success "Memory-safe Docker Compose created: docker-compose.gpu.safe.yml"

# Step 7: Memory diagnostic function
create_memory_diagnostic() {
    log_info "Creating memory diagnostic tools..."
    
    cat > diagnose_memory.sh << 'EOF'
#!/bin/bash
# Memory Corruption Diagnostic Tool

echo "Memory Corruption Diagnostic Report"
echo "=================================="
echo ""

echo "System Information:"
echo "-------------------"
free -h
echo ""

echo "Docker Information:"
echo "------------------"
docker --version
docker info | grep -E "(Runtime|Memory|Swap)"
echo ""

echo "NVIDIA Information:"
echo "------------------"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "NVIDIA drivers not found"
fi
echo ""

echo "Container Status:"
echo "----------------"
docker ps -a | grep whisper || echo "No whisper containers found"
echo ""

echo "Recent Container Logs:"
echo "---------------------"
container_id=$(docker ps -aq --filter "name=whisper" | head -1)
if [ ! -z "$container_id" ]; then
    docker logs --tail 50 "$container_id" 2>&1 | grep -E "(error|Error|ERROR|corruption|Corruption|CORRUPTION|segfault|abort|killed)"
else
    echo "No whisper containers found"
fi
echo ""

echo "Memory Usage by Process:"
echo "-----------------------"
ps aux --sort=-%mem | head -10
echo ""
EOF

    chmod +x diagnose_memory.sh
    log_success "Memory diagnostic tool created: ./diagnose_memory.sh"
}

create_memory_diagnostic

# Step 8: Build and test the memory-safe configuration
echo ""
read -p "Build and test the memory-safe configuration now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Building memory-safe image..."
    
    if docker-compose -f docker-compose.gpu.safe.yml build --no-cache; then
        log_success "Memory-safe image built successfully"
        
        log_info "Starting memory-safe application..."
        docker-compose -f docker-compose.gpu.safe.yml up -d
        
        log_info "Waiting for startup (30 seconds)..."
        sleep 30
        
        # Health check
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log_success "Application is running successfully!"
            echo ""
            echo "Access URL: http://localhost:8000"
            echo "Configuration: Memory-safe mode"
            echo "Model: Base (conservative)"
            echo "Batch size: 1 (safe)"
            echo ""
            echo "Monitoring:"
            echo "  Check logs: docker-compose -f docker-compose.gpu.safe.yml logs -f"
            echo "  Run diagnostics: ./diagnose_memory.sh"
            echo ""
        else
            log_warning "Health check failed. Run diagnostics with:"
            echo "  ./diagnose_memory.sh"
            echo "  docker-compose -f docker-compose.gpu.safe.yml logs"
        fi
    else
        log_error "Build failed. Check Docker logs for details."
        exit 1
    fi
fi

# Step 9: Instructions for recovery
echo ""
log_info "Memory Corruption Fix Applied"
echo ""
echo "Summary of changes:"
echo "  - Created memory-safe Docker configuration"
echo "  - Disabled JIT compilation completely"
echo "  - Limited to single-threaded processing"
echo "  - Conservative memory allocation"
echo "  - Stable library versions"
echo ""
echo "Usage:"
echo "  Start safe mode: docker-compose -f docker-compose.gpu.safe.yml up -d"
echo "  Stop containers: docker-compose -f docker-compose.gpu.safe.yml down"
echo "  View logs: docker-compose -f docker-compose.gpu.safe.yml logs -f"
echo "  Run diagnostics: ./diagnose_memory.sh"
echo ""
echo "Recovery:"
echo "  If issues persist, restore from backup: $backup_dir/"
echo "  For fallback to CPU: docker-compose -f docker-compose.cpu.yml up -d"
echo ""
echo "Next steps if memory corruption continues:"
echo "  1. Try smaller model (tiny instead of base)"
echo "  2. Reduce CUDA_MEMORY_FRACTION to 0.3"
echo "  3. Use CPU-only mode for stability"
echo "  4. Check system RAM (requires 8GB+ available)"
echo ""

log_success "Memory corruption fix completed!"
