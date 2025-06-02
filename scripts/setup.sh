#!/bin/bash
# Real-time Whisper Subtitles - Setup Script
# CUDA 12.9.0 + cuDNN optimized setup
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        log_success "$1 is installed"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "Checking GPU..."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    else
        log_warning "nvidia-smi not found - GPU support may not be available"
        return 1
    fi
}

# Check prerequisites
log_info "Checking prerequisites..."

# Check Docker
if ! check_command docker; then
    log_error "Docker is required but not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! check_command docker-compose; then
    log_error "Docker Compose is required but not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check GPU
check_gpu

# Create directories
log_info "Creating data directories..."
mkdir -p data/{models,outputs,logs,cache}
mkdir -p config/grafana/{dashboards,datasources}
mkdir -p config/nginx/ssl

# Set permissions
chmod -R 755 data/
chmod -R 755 config/

log_success "Directories created successfully"

# Copy environment file
log_info "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    log_success "Environment file created from .env.example"
    log_warning "Please review and modify .env file as needed"
else
    log_info "Environment file already exists"
fi

# Download models (optional)
read -p "Download Whisper models now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Downloading Whisper models..."
    docker run --rm -v "$(pwd)/data/models:/app/models" \
        python:3.11-slim python -c "
import subprocess
import sys

# Install whisper
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai-whisper'])

# Download models
import whisper
for model in ['tiny', 'base', 'small']:
    print(f'Downloading {model} model...')
    whisper.load_model(model, download_root='/app/models/whisper')
    print(f'{model} model downloaded successfully')
"
    log_success "Models downloaded successfully"
fi

# Build Docker image
read -p "Build Docker image now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Building Docker image..."
    docker-compose build --no-cache
    log_success "Docker image built successfully"
fi

# Setup complete
log_success "Setup completed successfully!"
echo
echo "Next steps:"
echo "1. Review and modify .env file if needed"
echo "2. Start the application: docker-compose up -d"
echo "3. Access the web interface: http://localhost:8000"
echo "4. For monitoring: docker-compose --profile monitoring up -d"
echo
echo "For more information, see README.md and SETUP.md"
