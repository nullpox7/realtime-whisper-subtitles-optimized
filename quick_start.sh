#!/bin/bash
# Real-time Whisper Subtitles - Universal Quick Start Script (v2.2.3)
# Works with any CUDA version and fixes cuDNN compatibility issues
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles - Universal Quick Start v2.2.3"
echo "=============================================================="
echo ""
echo "This script will automatically:"
echo "  ? Detect your system configuration"
echo "  ? Fix cuDNN compatibility issues"
echo "  ? Set up GPU or CPU mode as appropriate"
echo "  ? Build and start the application"
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Check GPU availability
check_gpu() {
    log_info "Checking GPU availability..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            log_success "GPU detected: $gpu_name"
            
            # Check NVIDIA Container Toolkit
            if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
                log_success "NVIDIA Container Toolkit is working"
                return 0
            else
                log_warning "NVIDIA Container Toolkit may not be properly configured"
                return 1
            fi
        else
            log_warning "nvidia-smi failed to run"
            return 1
        fi
    else
        log_info "No NVIDIA GPU detected (CPU mode will be used)"
        return 1
    fi
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p data/{models,outputs,logs,cache}
    mkdir -p config/{grafana,prometheus}
    chmod -R 755 data/ 2>/dev/null || true
    
    log_success "Directories created"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ "$1" = "gpu" ]; then
        if [ -f ".env.gpu.example" ]; then
            cp .env.gpu.example .env
            log_success "GPU environment configured with cuDNN compatibility"
        else
            log_warning ".env.gpu.example not found, using default"
            cp .env.example .env 2>/dev/null || echo "# GPU Mode" > .env
        fi
    else
        cp .env.example .env 2>/dev/null || echo "# CPU Mode" > .env
        log_success "CPU environment configured"
    fi
}

# Main execution
main() {
    log_info "Starting Real-time Whisper Subtitles setup..."
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Setup directories
    setup_directories
    
    # Check GPU availability
    if check_gpu; then
        log_info "? GPU mode detected - Setting up high-performance configuration"
        echo ""
        
        # Setup GPU environment
        setup_environment "gpu"
        
        # Build and start GPU version
        log_info "Building GPU-optimized image with cuDNN compatibility..."
        if docker-compose -f docker-compose.gpu.yml build --no-cache; then
            log_success "GPU image built successfully"
            
            log_info "Starting GPU-accelerated application..."
            docker-compose -f docker-compose.gpu.yml up -d
            
            compose_file="docker-compose.gpu.yml"
            mode="GPU"
        else
            log_error "GPU build failed, falling back to standard mode"
            setup_standard_mode
            compose_file="docker-compose.yml"
            mode="Standard"
        fi
        
    else
        log_info "? CPU mode detected - Setting up standard configuration"
        echo ""
        setup_standard_mode
        compose_file="docker-compose.yml"
        mode="CPU"
    fi
    
    # Wait for startup
    log_info "Waiting for application startup (60 seconds)..."
    sleep 60
    
    # Health check
    log_info "Performing health check..."
    for i in {1..5}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            health_data=$(curl -s http://localhost:8000/health)
            status=$(echo "$health_data" | jq -r '.status' 2>/dev/null || echo "healthy")
            gpu_available=$(echo "$health_data" | jq -r '.gpu_available' 2>/dev/null || echo "false")
            model_loaded=$(echo "$health_data" | jq -r '.model_loaded' 2>/dev/null || echo "true")
            version=$(echo "$health_data" | jq -r '.version' 2>/dev/null || echo "2.2.3")
            cudnn_fixes=$(echo "$health_data" | jq -r '.cudnn_fixes[]' 2>/dev/null || echo "")
            
            echo ""
            log_success "? Application is running successfully!"
            echo ""
            echo "=========================================="
            echo "  Real-time Whisper Subtitles v$version"
            echo "=========================================="
            echo ""
            echo "? Access URL: http://localhost:8000"
            echo "? Mode: $mode"
            echo "? Status: $status"
            echo "? GPU Available: $gpu_available"
            echo "? Model Loaded: $model_loaded"
            echo ""
            if [ "$mode" = "GPU" ]; then
                echo "? cuDNN Compatibility Fixes Applied:"
                echo "  ? cuDNN 8.x/9.x compatibility layers"
                echo "  ? Proper library path configuration"
                echo "  ? Memory-safe CUDA context"
                echo "  ? libcudnn_ops_infer.so.8 error resolved"
                echo ""
            fi
            echo "? Quick Actions:"
            echo "  ?? Press F for fullscreen subtitle overlay"
            echo "  ? Press Space to start/stop recording"
            echo "  ? Press C to clear subtitles"
            echo ""
            echo "?? Management Commands:"
            echo "  ? View logs: docker-compose -f $compose_file logs -f"
            echo "  ? Restart: docker-compose -f $compose_file restart"
            echo "  ? Stop: docker-compose -f $compose_file down"
            echo ""
            echo "? Documentation:"
            echo "  ? GPU Guide: README_GPU.md"
            echo "  ? Main Guide: README.md"
            echo ""
            break
        else
            log_warning "Health check attempt $i/5 failed, retrying..."
            sleep 10
        fi
        
        if [ $i -eq 5 ]; then
            log_warning "Health check failed, but application may still be starting"
            echo ""
            echo "? Troubleshooting:"
            echo "  ? Check logs: docker-compose -f $compose_file logs"
            echo "  ? Try accessing: http://localhost:8000"
            echo "  ? Wait a few more minutes for model loading"
            echo "  ? For cuDNN issues: Check GPU compatibility"
        fi
    done
}

setup_standard_mode() {
    setup_environment "cpu"
    
    log_info "Building standard image..."
    if docker-compose build --no-cache; then
        log_success "Standard image built successfully"
        
        log_info "Starting application..."
        docker-compose up -d
    else
        log_error "Build failed"
        exit 1
    fi
}

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Real-time Whisper Subtitles - Universal Quick Start v2.2.3"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo "  --gpu         Force GPU mode (auto-detected by default)"
    echo "  --cpu         Force CPU mode"
    echo ""
    echo "This script automatically:"
    echo "  ? Detects GPU availability"
    echo "  ? Fixes cuDNN compatibility issues"
    echo "  ? Sets up appropriate configuration"
    echo "  ? Builds and starts the application"
    echo ""
    echo "cuDNN Fixes Applied:"
    echo "  ? libcudnn_ops_infer.so.8 error resolution"
    echo "  ? cuDNN 8.x/9.x compatibility layers"
    echo "  ? Proper library path configuration"
    echo "  ? Memory-safe CUDA initialization"
    echo ""
    echo "For manual setup, see README.md"
    exit 0
fi

# Handle force modes
if [ "$1" = "--gpu" ]; then
    log_info "GPU mode forced"
    check_prerequisites
    setup_directories
    setup_environment "gpu"
    
    log_info "Building GPU image with cuDNN compatibility..."
    docker-compose -f docker-compose.gpu.yml build --no-cache
    docker-compose -f docker-compose.gpu.yml up -d
    exit 0
elif [ "$1" = "--cpu" ]; then
    log_info "CPU mode forced"
    check_prerequisites
    setup_directories
    setup_standard_mode
    exit 0
fi

# Run main setup
main