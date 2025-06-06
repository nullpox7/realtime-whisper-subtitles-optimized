#!/bin/bash
# Real-time Whisper Subtitles - ABI?????????? (v2.2.2)
# PyTorch/torchaudio ABI?????????????
# Encoding: UTF-8

set -e

echo "? PyTorch/torchaudio ABI?????????? v2.2.2"
echo "========================================================="
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

# Step 1: Stop and clean containers
cleanup_containers() {
    log_info "????????????????..."
    
    # Stop all related containers
    docker-compose -f docker-compose.gpu.yml down --remove-orphans 2>/dev/null || true
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove any existing containers
    docker container prune -f >/dev/null 2>&1 || true
    
    log_success "?????????????"
}

# Step 2: Clean Docker build cache
clean_build_cache() {
    log_info "Docker ????????????????..."
    
    # Remove specific images
    docker images | grep realtime-whisper-subtitles | awk '{print $3}' | xargs -r docker rmi -f >/dev/null 2>&1 || true
    
    # Clean build cache
    docker builder prune -f >/dev/null 2>&1 || true
    
    log_success "????????????????"
}

# Step 3: Check NVIDIA GPU availability
check_gpu() {
    log_info "GPU?????????..."
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi ?????????NVIDIA ???????????????????????????"
        return 1
    fi
    
    if ! nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi ???????????GPU ???????????????????????"
        return 1
    fi
    
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")
    
    log_success "GPU??: $gpu_name ($gpu_memory, CUDA $cuda_version)"
    
    # Check NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        log_warning "NVIDIA Container Toolkit ????????????????????"
        log_info "NVIDIA Container Toolkit ???????/????????"
        return 1
    fi
    
    log_success "NVIDIA Container Toolkit ???????????"
    return 0
}

# Step 4: Run PyTorch diagnostic in container
run_diagnostic() {
    log_info "PyTorch ?????????..."
    
    # Create temporary diagnostic script
    cat > /tmp/pytorch_diagnostic.py << 'EOF'
#!/usr/bin/env python3
import sys
try:
    print("? Python version:", sys.version)
    
    import torch
    print(f"? PyTorch version: {torch.__version__}")
    print(f"? CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"? CUDA version: {torch.version.cuda}")
        print(f"? GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"? GPU name: {torch.cuda.get_device_name(0)}")
    
    import torchaudio
    print(f"? torchaudio version: {torchaudio.__version__}")
    
    # Test basic functionality
    print("? Testing basic functionality...")
    tensor = torch.randn(2, 3)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        print("? CUDA tensor creation successful")
    
    print("? All tests passed!")
    
except ImportError as e:
    print(f"? Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"? Test failed: {e}")
    sys.exit(1)
EOF

    # Run diagnostic in container
    if docker run --rm --gpus all \
        -v /tmp/pytorch_diagnostic.py:/tmp/diagnostic.py \
        nvidia/cuda:12.4.1-devel-ubuntu22.04 \
        bash -c "
            apt-get update >/dev/null 2>&1 && 
            apt-get install -y python3.11 python3-pip curl >/dev/null 2>&1 && 
            curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 >/dev/null 2>&1 &&
            python3.11 -m pip install --quiet torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124 && 
            python3.11 /tmp/diagnostic.py
        " 2>/dev/null; then
        log_success "PyTorch ???????"
        rm -f /tmp/pytorch_diagnostic.py
        return 0
    else
        log_error "PyTorch ???????"
        rm -f /tmp/pytorch_diagnostic.py
        return 1
    fi
}

# Step 5: Build GPU image with fixed dependencies
build_gpu_image() {
    log_info "???GPU Docker?????????..."
    
    if docker-compose -f docker-compose.gpu.yml build --no-cache whisper-subtitles-gpu; then
        log_success "GPU Docker?????????"
        return 0
    else
        log_error "GPU Docker?????????"
        return 1
    fi
}

# Step 6: Test the built image
test_built_image() {
    log_info "???????????????..."
    
    # Get the image name
    image_name=$(docker-compose -f docker-compose.gpu.yml images -q whisper-subtitles-gpu 2>/dev/null || echo "")
    
    if [ -z "$image_name" ]; then
        # Fallback to compose project name
        project_name=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
        image_name="${project_name}-whisper-subtitles-gpu"
    fi
    
    log_info "?????????: $image_name"
    
    # Test the image
    if docker run --rm --gpus all \
        -e HOST=0.0.0.0 \
        -e PORT=8000 \
        -e WHISPER_MODEL=tiny \
        "$image_name" \
        timeout 30 python3.11 -c "
import torch
import torchaudio
import sys
print('? PyTorch version:', torch.__version__)
print('? torchaudio version:', torchaudio.__version__)
print('? CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('? GPU count:', torch.cuda.device_count())
print('? All imports successful!')
sys.exit(0)
        " 2>/dev/null; then
        log_success "?????????"
        return 0
    else
        log_error "?????????"
        return 1
    fi
}

# Step 7: Start services
start_services() {
    log_info "GPU????????..."
    
    if docker-compose -f docker-compose.gpu.yml up -d; then
        log_success "????????"
        
        # Wait for startup
        log_info "?????????? (60?)..."
        sleep 60
        
        # Health check
        log_info "??????????..."
        for i in {1..5}; do
            if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                health_data=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{}')
                status=$(echo "$health_data" | jq -r '.status' 2>/dev/null || echo "unknown")
                gpu_available=$(echo "$health_data" | jq -r '.gpu_available' 2>/dev/null || echo "unknown")
                model_loaded=$(echo "$health_data" | jq -r '.model_loaded' 2>/dev/null || echo "unknown")
                version=$(echo "$health_data" | jq -r '.version' 2>/dev/null || echo "unknown")
                
                echo ""
                log_success "? ????????????????????????"
                echo ""
                echo "===========================================" 
                echo "  Real-time Whisper Subtitles (???)"
                echo "==========================================="
                echo ""
                echo "? ????URL: http://localhost:8000"
                echo "? ?????: $status"
                echo "? GPU????: $gpu_available"
                echo "? ?????????: $model_loaded"
                echo "? ?????: $version"
                echo ""
                echo "? ABI??????????:"
                echo "  ? PyTorch 2.4.1+cu124 (???????)"
                echo "  ? torchaudio 2.4.1+cu124 (???????)"
                echo "  ? ????????"
                echo "  ? ABI?????"
                echo ""
                echo "? ????:"
                echo "  ? F ??: ??????????"
                echo "  ?? Space ??: ????/??"
                echo "  ? C ??: ?????"
                echo ""
                echo "? ??????:"
                echo "  ? ????: docker-compose -f docker-compose.gpu.yml logs -f"
                echo "  ? ???: docker-compose -f docker-compose.gpu.yml restart"
                echo "  ?? ??: docker-compose -f docker-compose.gpu.yml down"
                echo "  ? GPU??: watch -n 1 nvidia-smi"
                echo ""
                return 0
            else
                log_warning "????????? $i/5 ???????..."
                sleep 10
            fi
        done
        
        log_warning "???????????????????????????????????????"
        echo ""
        echo "? ???????????:"
        echo "  ? ????: docker-compose -f docker-compose.gpu.yml logs whisper-subtitles-gpu"
        echo "  ? ??????: http://localhost:8000"
        echo "  ? ??????????????????"
        return 1
        
    else
        log_error "????????"
        return 1
    fi
}

# Step 8: Show troubleshooting info
show_troubleshooting() {
    echo ""
    log_info "? ?????????????"
    echo "=================================="
    echo ""
    echo "? ??????????"
    echo ""
    echo "1. ? ?????????:"
    echo "   docker-compose -f docker-compose.gpu.yml down --volumes --remove-orphans"
    echo "   docker system prune -af"
    echo "   docker volume prune -f"
    echo ""
    echo "2. ? ???:"
    echo "   ./fix_abi_compatibility.sh"
    echo ""
    echo "3. ? ????:"
    echo "   docker-compose -f docker-compose.gpu.yml logs whisper-subtitles-gpu"
    echo ""
    echo "4. ? ????:"
    echo "   python3 pytorch_diagnostic.py  # ??????????"
    echo ""
    echo "5. ? CPU???????:"
    echo "   docker-compose -f docker-compose.cpu.yml up -d"
    echo ""
    echo "6. ? ????:"
    echo "   GitHub Issues: https://github.com/nullpox7/realtime-whisper-subtitles-optimized/issues"
    echo ""
}

# Main execution
main() {
    echo "? ABI???????????????..."
    echo ""
    
    # Step 1: Cleanup
    cleanup_containers
    
    # Step 2: Clean cache
    clean_build_cache
    
    # Step 3: Check GPU
    if ! check_gpu; then
        log_error "GPU ???????????"
        log_info "CPU????????????????: docker-compose -f docker-compose.cpu.yml up -d"
        exit 1
    fi
    
    # Step 4: Run diagnostic
    if ! run_diagnostic; then
        log_warning "???????????????????..."
    fi
    
    # Step 5: Build image
    if ! build_gpu_image; then
        log_error "Docker???????????????"
        show_troubleshooting
        exit 1
    fi
    
    # Step 6: Test image
    if ! test_built_image; then
        log_warning "?????????????????????..."
    fi
    
    # Step 7: Start services
    if start_services; then
        log_success "? ABI?????????????"
    else
        log_error "?????????????"
        show_troubleshooting
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "ABI??????????"
        echo ""
        echo "????: $0 [?????]"
        echo ""
        echo "?????:"
        echo "  --help, -h     ?????????????"
        echo "  --clean-only   ???????????"
        echo "  --test-only    ?????????"
        echo "  --build-only   ???????"
        echo ""
        echo "????????????????:"
        echo "  1. ??????????????"
        echo "  2. Docker ???????????????"
        echo "  3. GPU ??????"
        echo "  4. PyTorch ??????"
        echo "  5. ???Docker????????"
        echo "  6. ????????"
        echo "  7. ???????"
        echo ""
        exit 0
        ;;
    --clean-only)
        log_info "????????????..."
        cleanup_containers
        clean_build_cache
        log_success "?????????"
        exit 0
        ;;
    --test-only)
        log_info "??????????..."
        check_gpu
        run_diagnostic
        exit 0
        ;;
    --build-only)
        log_info "????????..."
        build_gpu_image
        test_built_image
        exit 0
        ;;
    "")
        # Default: run full process
        main
        ;;
    *)
        log_error "????????: $1"
        echo "??????????: $0 --help"
        exit 1
        ;;
esac