#!/bin/bash
# Docker Build Error Fix Script
# Automatically detects and fixes common Docker build issues
# Encoding: UTF-8

set -e

echo "? Docker Build Error Fix Script"
echo "================================="
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

# Function to check build error type
analyze_build_error() {
    local dockerfile="$1"
    log_info "Analyzing build errors for: $dockerfile"
    
    # Try to build and capture error
    if docker build -f "$dockerfile" . 2>&1 | grep -q "apt-get.*exit code: 100"; then
        log_error "Detected apt-get package installation failure"
        return 1
    elif docker build -f "$dockerfile" . 2>&1 | grep -q "python3.11.*not found"; then
        log_error "Detected Python 3.11 installation issue"
        return 2
    elif docker build -f "$dockerfile" . 2>&1 | grep -q "libnccl.*not found"; then
        log_error "Detected NCCL library issue"
        return 3
    else
        log_info "No specific error pattern detected"
        return 0
    fi
}

# Function to fix apt-get issues
fix_apt_issues() {
    log_info "Creating fixed Dockerfile for apt-get issues..."
    
    cat > Dockerfile.gpu.build-fixed << 'EOF'
# Real-time Whisper Subtitles - Build Error Fixed Dockerfile
# Robust package installation with error handling
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LOG_PATH=/app/data/logs

# Update package lists with retry
RUN apt-get update || (sleep 5 && apt-get update)

# Install packages in stages to isolate issues
RUN apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-distutils \
    && apt-get clean

RUN apt-get install -y --no-install-recommends \
    build-essential curl wget git \
    && apt-get clean

RUN apt-get install -y --no-install-recommends \
    ffmpeg libavcodec-dev libavformat-dev \
    && apt-get clean || echo "Some audio packages not available, continuing..."

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install Python packages
COPY requirements.gpu.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir -r requirements.gpu.txt

# Setup directories and files
RUN mkdir -p /app/data/{models,outputs,logs,cache} /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser

COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY .env.gpu.example* ./

RUN chown -R appuser:appuser /app

USER appuser

RUN mkdir -p /app/data/{models/whisper,outputs,logs,cache}

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL=base
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16

CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

    log_success "Created: Dockerfile.gpu.build-fixed"
}

# Function to test build
test_build() {
    local dockerfile="$1"
    log_info "Testing build with: $dockerfile"
    
    if docker build -f "$dockerfile" -t whisper-test . >/dev/null 2>&1; then
        log_success "Build successful with $dockerfile"
        return 0
    else
        log_warning "Build failed with $dockerfile"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting Docker build error diagnosis..."
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Array of Dockerfiles to try, in order of preference
    DOCKERFILES=(
        "Dockerfile.gpu.lite"
        "Dockerfile.gpu.fixed" 
        "Dockerfile.gpu.stable"
        "Dockerfile.gpu"
    )
    
    # Test each Dockerfile
    working_dockerfile=""
    for dockerfile in "${DOCKERFILES[@]}"; do
        if [ -f "$dockerfile" ]; then
            log_info "Testing: $dockerfile"
            if test_build "$dockerfile"; then
                working_dockerfile="$dockerfile"
                break
            fi
        else
            log_warning "File not found: $dockerfile"
        fi
    done
    
    # If none work, create a fixed version
    if [ -z "$working_dockerfile" ]; then
        log_warning "All existing Dockerfiles failed, creating fixed version..."
        fix_apt_issues
        
        if test_build "Dockerfile.gpu.build-fixed"; then
            working_dockerfile="Dockerfile.gpu.build-fixed"
        else
            log_error "Even the fixed Dockerfile failed to build"
            echo ""
            log_info "Manual troubleshooting steps:"
            echo "1. Check internet connection"
            echo "2. Try: docker system prune -a"
            echo "3. Update Docker: docker version"
            echo "4. Check available disk space: df -h"
            echo "5. Try building with verbose output:"
            echo "   docker build -f Dockerfile.gpu.lite --progress=plain ."
            exit 1
        fi
    fi
    
    log_success "Found working Dockerfile: $working_dockerfile"
    echo ""
    
    # Update docker-compose.yml to use the working Dockerfile
    if [ -f "docker-compose.gpu.yml" ]; then
        log_info "Updating docker-compose.gpu.yml..."
        
        # Create backup
        cp docker-compose.gpu.yml docker-compose.gpu.yml.backup
        
        # Update dockerfile reference
        sed -i "s/dockerfile: Dockerfile\.gpu.*/dockerfile: $working_dockerfile/" docker-compose.gpu.yml
        
        log_success "Updated docker-compose.gpu.yml to use: $working_dockerfile"
        echo ""
    fi
    
    # Offer to build the complete application
    read -p "Build the complete application now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Building complete application..."
        
        if docker-compose -f docker-compose.gpu.yml build --no-cache; then
            log_success "? Build completed successfully!"
            echo ""
            
            read -p "Start the application? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Starting application..."
                docker-compose -f docker-compose.gpu.yml up -d
                
                log_info "Waiting for startup (30 seconds)..."
                sleep 30
                
                if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                    log_success "? Application is running!"
                    echo "Access at: http://localhost:8000"
                else
                    log_warning "Application started but health check failed"
                    echo "Check logs: docker-compose -f docker-compose.gpu.yml logs"
                fi
            fi
        else
            log_error "Build failed even with working Dockerfile"
            echo "Check the build output above for specific errors"
        fi
    fi
    
    echo ""
    log_success "Build error fix completed!"
    echo ""
    echo "? Summary:"
    echo "- Working Dockerfile: $working_dockerfile"
    echo "- Updated: docker-compose.gpu.yml"
    echo "- Backup: docker-compose.gpu.yml.backup"
    echo ""
    echo "? Manual build command:"
    echo "docker build -f $working_dockerfile -t whisper-gpu ."
}

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Docker Build Error Fix Script"
    echo ""
    echo "Usage: $0 [dockerfile]"
    echo ""
    echo "This script automatically:"
    echo "  ? Detects Docker build errors"
    echo "  ? Tests multiple Dockerfile variants"
    echo "  ? Creates fixed versions when needed"
    echo "  ? Updates docker-compose configuration"
    echo ""
    echo "Options:"
    echo "  dockerfile    Specific Dockerfile to test (optional)"
    echo "  --help, -h    Show this help message"
    echo ""
    exit 0
fi

# If specific dockerfile provided, test only that one
if [ -n "$1" ] && [ -f "$1" ]; then
    log_info "Testing specific Dockerfile: $1"
    if test_build "$1"; then
        log_success "? $1 builds successfully"
    else
        log_error "? $1 failed to build"
    fi
    exit 0
fi

# Run main function
main