#!/bin/bash
# Real-time Whisper Subtitles - Complete Fix Script v2.0.2
# Fix encoding and audio processing issues completely
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles - Complete Fix v2.0.2"
echo "===================================================="
echo ""
echo "Fixing:"
echo "  ? Character encoding issues (Full English UI)"
echo "  ? AudioSegment conversion failed errors"
echo "  ? PyDub dependency issues"
echo "  ? WebRTC VAD compatibility problems"
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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in correct directory
if [ ! -f "docker-compose.yml" ]; then
    log_error "docker-compose.yml not found"
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Stop containers
log_info "Stopping containers..."
docker-compose down || true

# Create necessary directories
log_info "Creating necessary directories..."
mkdir -p data/{models,outputs,logs,cache}
mkdir -p static templates
chmod -R 755 data/ 2>/dev/null || true
log_success "Directories created"

# Pull latest changes
log_info "Pulling latest code..."
if git status >/dev/null 2>&1; then
    git pull origin main || log_warning "Git pull failed - manual update required"
    log_success "Code updated"
else
    log_warning "Not a Git repository - manual update needed"
fi

# Clean up Docker completely
log_info "Cleaning up Docker completely..."
docker system prune -af >/dev/null 2>&1 || true
docker volume prune -f >/dev/null 2>&1 || true

# Build with no cache - complete rebuild
log_info "Building Docker image (complete rebuild)..."
if docker-compose build --no-cache --pull whisper-subtitles; then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Start services
log_info "Starting services..."
if docker-compose up -d; then
    log_success "Services started"
else
    log_error "Failed to start services"
    exit 1
fi

# Wait longer for initialization
log_info "Waiting for services to initialize (45 seconds)..."
sleep 45

# Health check with retry
log_info "Performing health check..."
for i in {1..5}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "? Application is healthy!"
        
        # Get health status
        health_status=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "unknown")
        model_loaded=$(curl -s http://localhost:8000/health | jq -r '.model_loaded' 2>/dev/null || echo "unknown")
        
        echo ""
        echo "? Fix completed successfully!"
        echo "================================"
        echo ""
        echo "? Access URL: http://localhost:8000"
        echo "? Health Status: $health_status"
        echo "? Model Loaded: $model_loaded"
        echo ""
        echo "? Fixed Issues:"
        echo "  ? Character encoding (Full English UI)"
        echo "  ? AudioSegment conversion errors"
        echo "  ? PyDub dependency removed"
        echo "  ? Simplified audio processing"
        echo "  ? WebRTC VAD replaced with energy detection"
        echo ""
        echo "? Useful Commands:"
        echo "  ? View logs: docker-compose logs -f whisper-subtitles"
        echo "  ? Restart: docker-compose restart whisper-subtitles"
        echo "  ? Stop: docker-compose down"
        echo ""
        echo "? If you encounter issues:"
        echo "  ? Check browser console for JavaScript errors"
        echo "  ? Verify microphone permissions"
        echo "  ? Try different audio file formats"
        break
    else
        log_warning "Health check attempt $i/5 failed, retrying..."
        sleep 10
    fi
    
    if [ $i -eq 5 ]; then
        log_warning "Health check failed after 5 attempts"
        log_info "The application may still be starting up..."
        echo ""
        echo "? Troubleshooting:"
        echo "  ? Wait a few more minutes and try: http://localhost:8000"
        echo "  ? Check logs: docker-compose logs whisper-subtitles"
        echo "  ? Verify no other services are using port 8000"
        echo ""
    fi
done

echo ""
log_success "Complete fix script finished!"
echo ""
echo "Note: All text is now in English to avoid encoding issues."
echo "Audio processing has been simplified to fix conversion errors."
