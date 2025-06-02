#!/bin/bash
# Real-time Whisper Subtitles - Cleanup Script
# Clean up Docker containers, images, and data
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles Cleanup"
echo "====================================="

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

# Stop all containers
log_info "Stopping Docker containers..."
docker-compose down --remove-orphans || true
docker-compose --profile monitoring down --remove-orphans || true
log_success "Containers stopped"

# Remove containers
log_info "Removing containers..."
docker container prune -f
log_success "Containers cleaned up"

# Clean up images (optional)
read -p "Remove Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Removing Docker images..."
    docker image prune -f
    docker images | grep realtime-whisper-subtitles | awk '{print $3}' | xargs -r docker rmi -f
    log_success "Images cleaned up"
fi

# Clean up volumes (optional)
read -p "Remove Docker volumes? This will delete all data! (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_warning "This will delete all persistent data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing Docker volumes..."
        docker volume prune -f
        log_success "Volumes cleaned up"
    fi
fi

# Clean up logs (optional)
read -p "Clean up log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Cleaning up log files..."
    rm -rf data/logs/*.log
    rm -rf data/outputs/*
    rm -rf data/cache/*
    log_success "Log files cleaned up"
fi

# Clean up Python cache
log_info "Cleaning up Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
log_success "Python cache cleaned up"

log_success "Cleanup completed!"
echo "To start fresh, run: ./scripts/setup.sh"
