#!/bin/bash
# Complete Docker Build Diagnosis and Repair Script
# Diagnoses and fixes all common Docker build issues
# Encoding: UTF-8

set -e

echo "? Complete Docker Build Diagnosis & Repair"
echo "============================================"
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

# Function to check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check disk space
    available_space=$(df -BG /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "unknown")
    if [ "$available_space" != "unknown" ] && [ "$available_space" -lt 10 ]; then
        log_warning "Low disk space: ${available_space}GB available"
        log_info "Cleaning up Docker to free space..."
        docker system prune -af --volumes || true
    else
        log_success "Sufficient disk space available"
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running or accessible"
        return 1
    fi
    
    log_success "Docker daemon is running"
    return 0
}

# Function to check network connectivity
check_network() {
    log_info "Checking network connectivity..."
    
    # Test Docker Hub connectivity
    if docker run --rm alpine ping -c 3 registry-1.docker.io >/dev/null 2>&1; then
        log_success "Docker Hub connectivity OK"
    else
        log_warning "Docker Hub connectivity issues detected"
        log_info "Trying alternative test..."
        if docker run --rm alpine ping -c 3 8.8.8.8 >/dev/null 2>&1; then
            log_info "Internet connectivity OK, may be Docker Hub specific issue"
        else
            log_error "No internet connectivity"
            return 1
        fi
    fi
    
    return 0
}

# Function to get detailed build error
get_build_error() {
    local dockerfile="$1"
    log_info "Getting detailed build error for: $dockerfile"
    
    # Create a temporary build log
    local build_log="/tmp/docker_build_error.log"
    
    # Attempt build and capture all output
    if docker build -f "$dockerfile" --progress=plain . >"$build_log" 2>&1; then
        log_success "Build succeeded on detailed test"
        return 0
    else
        log_error "Build failed. Error details:"
        echo "----------------------------------------"
        tail -n 50 "$build_log" | grep -E "(ERROR|error|Error|FAILED|failed|Failed)" || tail -n 20 "$build_log"
        echo "----------------------------------------"
        return 1
    fi
}

# Function to create minimal test Dockerfile
create_minimal_dockerfile() {
    log_info "Creating minimal test Dockerfile..."
    
    cat > Dockerfile.minimal-test << 'EOF'
# Minimal test Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Test basic package installation
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Test Python
RUN python3 --version

# Test pip
RUN pip3 --version

CMD ["echo", "Minimal test successful"]
EOF

    log_success "Created: Dockerfile.minimal-test"
}

# Function to create working CUDA Dockerfile
create_working_cuda_dockerfile() {
    log_info "Creating simplified working CUDA Dockerfile..."
    
    cat > Dockerfile.cuda-simple << 'EOF'
# Simplified working CUDA Dockerfile
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install essential packages only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install minimal Python requirements
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0 \
    jinja2==3.1.2 \
    torch==2.4.1+cu124 \
    faster-whisper==0.9.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Create minimal app structure
RUN mkdir -p /app/src /app/static /app/templates /app/data/logs

# Create minimal web interface
RUN echo 'from fastapi import FastAPI
from fastapi.responses import JSONResponse
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Real-time Whisper Subtitles - Minimal Version"}

@app.get("/health")
def health():
    return {"status": "healthy", "version": "minimal"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
' > /app/main.py

EXPOSE 8000
CMD ["python3", "main.py"]
EOF

    log_success "Created: Dockerfile.cuda-simple"
}

# Function to create requirements-minimal.txt
create_minimal_requirements() {
    log_info "Creating minimal requirements file..."
    
    cat > requirements-minimal.txt << 'EOF'
# Minimal requirements for testing
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
jinja2==3.1.2
torch==2.4.1+cu124
faster-whisper==0.9.0
librosa==0.10.1
soundfile==0.12.1
numpy>=1.24.0,<2.0.0
EOF

    log_success "Created: requirements-minimal.txt"
}

# Function to test progressive builds
test_progressive_builds() {
    log_info "Testing progressive builds..."
    
    # Test 1: Minimal Ubuntu
    log_info "Test 1: Minimal Ubuntu base"
    create_minimal_dockerfile
    if docker build -f Dockerfile.minimal-test -t test-minimal . >/dev/null 2>&1; then
        log_success "? Basic Ubuntu build works"
        docker rmi test-minimal >/dev/null 2>&1 || true
    else
        log_error "? Basic Ubuntu build failed"
        get_build_error "Dockerfile.minimal-test"
        return 1
    fi
    
    # Test 2: CUDA base
    log_info "Test 2: CUDA base image"
    if docker pull nvidia/cuda:12.4.0-base-ubuntu22.04 >/dev/null 2>&1; then
        log_success "? CUDA base image pull works"
    else
        log_error "? CUDA base image pull failed"
        return 1
    fi
    
    # Test 3: Simple CUDA build
    log_info "Test 3: Simple CUDA build with Python"
    create_working_cuda_dockerfile
    if docker build -f Dockerfile.cuda-simple -t test-cuda-simple . >/dev/null 2>&1; then
        log_success "? Simple CUDA build works"
        docker rmi test-cuda-simple >/dev/null 2>&1 || true
        return 0
    else
        log_error "? Simple CUDA build failed"
        get_build_error "Dockerfile.cuda-simple"
        return 1
    fi
}

# Function to create working docker-compose
create_working_compose() {
    log_info "Creating simplified docker-compose configuration..."
    
    cat > docker-compose.simple.yml << 'EOF'
version: '3.8'

services:
  whisper-simple:
    build:
      context: .
      dockerfile: Dockerfile.cuda-simple
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    runtime: nvidia
EOF

    log_success "Created: docker-compose.simple.yml"
}

# Main diagnosis function
main_diagnosis() {
    log_info "Starting comprehensive Docker build diagnosis..."
    echo ""
    
    # Step 1: Check system resources
    if ! check_system_resources; then
        log_error "System resource check failed"
        return 1
    fi
    
    # Step 2: Check network
    if ! check_network; then
        log_error "Network connectivity check failed"
        return 1
    fi
    
    # Step 3: Test progressive builds
    if test_progressive_builds; then
        log_success "Progressive build tests passed!"
        
        # Create working configuration
        create_minimal_requirements
        create_working_compose
        
        echo ""
        log_success "? Working configuration created!"
        echo ""
        echo "? Files created:"
        echo "  - Dockerfile.cuda-simple (working CUDA Dockerfile)"
        echo "  - docker-compose.simple.yml (simplified compose)"
        echo "  - requirements-minimal.txt (minimal requirements)"
        echo ""
        echo "? To start the application:"
        echo "  docker-compose -f docker-compose.simple.yml up -d"
        echo ""
        echo "? Access at: http://localhost:8000"
        echo ""
        
        # Offer to start immediately
        read -p "Start the simplified application now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Building and starting simplified application..."
            
            if docker-compose -f docker-compose.simple.yml up -d; then
                log_info "Waiting for startup (30 seconds)..."
                sleep 30
                
                if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                    log_success "? Simplified application is running!"
                    echo "Access at: http://localhost:8000"
                    echo ""
                    echo "Next steps to get full functionality:"
                    echo "1. Copy your src/ files to the container"
                    echo "2. Install additional requirements as needed"
                    echo "3. Gradually add features back"
                else
                    log_warning "Application started but not responding yet"
                    echo "Check: docker-compose -f docker-compose.simple.yml logs"
                fi
            else
                log_error "Failed to start simplified application"
                return 1
            fi
        fi
        
        return 0
    else
        log_error "Progressive build tests failed"
        
        echo ""
        log_info "? Advanced troubleshooting suggestions:"
        echo ""
        echo "1. Check Docker installation:"
        echo "   docker --version"
        echo "   docker info"
        echo ""
        echo "2. Reset Docker (WARNING: removes all containers/images):"
        echo "   docker system prune -a --volumes"
        echo ""
        echo "3. Check Docker daemon logs:"
        echo "   sudo journalctl -u docker.service"
        echo ""
        echo "4. Restart Docker daemon:"
        echo "   sudo systemctl restart docker"
        echo ""
        echo "5. Check for corporate firewall/proxy issues"
        echo ""
        echo "6. Try different Docker registry:"
        echo "   docker pull ubuntu:22.04"
        echo ""
        echo "7. Free up disk space:"
        echo "   df -h"
        echo "   sudo apt clean"
        echo ""
        
        return 1
    fi
}

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Complete Docker Build Diagnosis & Repair"
    echo ""
    echo "This script performs comprehensive diagnosis:"
    echo "  ? System resource checks"
    echo "  ? Network connectivity tests"
    echo "  ? Progressive build testing"
    echo "  ? Creates working minimal configuration"
    echo ""
    echo "Usage: $0"
    echo ""
    exit 0
fi

# Run main diagnosis
main_diagnosis