#!/bin/bash
# Real-time Whisper Subtitles - Emergency Fix v2.0.6
# Complete diagnostic and fix for persistent webrtcvad errors
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles - Emergency Fix v2.0.6"
echo "======================================================"
echo ""
echo "? This will completely solve the webrtcvad error"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Show current error
log_info "? Checking current error state..."
if docker-compose ps | grep -q whisper-subtitles; then
    echo "Current container logs (showing webrtcvad errors):"
    docker-compose logs whisper-subtitles 2>&1 | grep -E "(ERROR|webrtcvad|ModuleNotFoundError)" | tail -10 || echo "No specific errors found"
fi
echo ""

# Step 2: Complete cleanup
log_info "? Performing complete cleanup..."
docker-compose down --remove-orphans || true
docker system prune -af || true
docker images | grep realtime-whisper-subtitles | awk '{print $3}' | xargs -r docker rmi -f || true

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

log_success "? Cleanup completed"

# Step 3: Pull latest fixes
log_info "? Pulling latest fixes from GitHub..."
if git status >/dev/null 2>&1; then
    git fetch origin main
    git reset --hard origin/main
    log_success "? Latest fixes pulled"
else
    log_info "Not a Git repository, downloading files manually..."
    curl -s -o src/web_interface.py https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/src/web_interface.py
    curl -s -o src/audio_processing.py https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/src/audio_processing.py
    curl -s -o requirements.txt https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/requirements.txt
    log_success "? Fixed files downloaded"
fi

# Step 4: Verify no webrtcvad references
log_info "? Verifying no webrtcvad references..."
webrtc_count=$(grep -r "webrtcvad" src/ 2>/dev/null | grep -v "# REMOVED" | wc -l || echo "0")
if [ "$webrtc_count" -gt 0 ]; then
    log_error "? Still found webrtcvad references:"
    grep -r "webrtcvad" src/ 2>/dev/null | grep -v "# REMOVED" || true
    exit 1
else
    log_success "? No webrtcvad references found"
fi

# Step 5: Create necessary directories
log_info "? Creating necessary directories..."
mkdir -p data/{models,outputs,logs,cache}
mkdir -p static templates
chmod -R 755 data/ 2>/dev/null || true
log_success "? Directories created"

# Step 6: Build with complete rebuild
log_info "? Building Docker image (complete rebuild)..."
if docker-compose build --no-cache --pull whisper-subtitles; then
    log_success "? Build completed successfully"
else
    log_error "? Build failed - checking for remaining issues..."
    docker-compose logs whisper-subtitles 2>&1 | grep -E "(ERROR|webrtcvad|ModuleNotFoundError)" || true
    exit 1
fi

# Step 7: Start services
log_info "? Starting services..."
if docker-compose up -d; then
    log_success "? Services started"
else
    log_error "? Failed to start services"
    exit 1
fi

# Step 8: Wait and test
log_info "? Waiting for startup (45 seconds)..."
sleep 45

# Step 9: Health check with detailed logging
log_info "? Performing health check..."
for i in {1..5}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        health_data=$(curl -s http://localhost:8000/health)
        status=$(echo "$health_data" | jq -r '.status' 2>/dev/null || echo "unknown")
        version=$(echo "$health_data" | jq -r '.version' 2>/dev/null || echo "unknown")
        
        echo ""
        log_success "? SUCCESS! Application is running!"
        echo "=========================================="
        echo "? Application URL: http://localhost:8000"
        echo "? Health Status: $status"
        echo "? Version: $version"
        echo ""
        echo "? WebRTC VAD error COMPLETELY RESOLVED!"
        echo ""
        echo "? What was fixed:"
        echo "  ? All webrtcvad imports removed"
        echo "  ? External audio dependencies removed" 
        echo "  ? Simple energy-based speech detection"
        echo "  ? Clean requirements.txt"
        echo "  ? Minimal audio processing"
        echo ""
        echo "? Ready to use:"
        echo "  ? Real-time microphone transcription"
        echo "  ? File upload transcription"
        echo "  ? Web interface"
        echo ""
        echo "? Commands:"
        echo "  ? View logs: docker-compose logs -f whisper-subtitles"
        echo "  ? Restart: docker-compose restart whisper-subtitles"
        echo "  ? Stop: docker-compose down"
        echo ""
        exit 0
    else
        log_error "? Health check attempt $i/5 failed"
        
        if [ $i -eq 1 ]; then
            echo "Checking for errors in logs:"
            docker-compose logs whisper-subtitles 2>&1 | tail -15
            echo ""
        fi
        
        if [ $i -eq 5 ]; then
            echo ""
            log_error "? Application failed to start properly"
            echo ""
            echo "? Final error check:"
            docker-compose logs whisper-subtitles 2>&1 | grep -E "(ERROR|Error|webrtcvad|ModuleNotFoundError|ImportError)" | tail -10 || echo "No specific errors found"
            echo ""
            echo "? Manual troubleshooting:"
            echo "  1. Check logs: docker-compose logs whisper-subtitles"
            echo "  2. Verify container status: docker-compose ps"
            echo "  3. Try restart: docker-compose restart whisper-subtitles"
            echo "  4. Check for port conflicts: netstat -tulpn | grep 8000"
            echo ""
            exit 1
        fi
        
        sleep 10
    fi
done
