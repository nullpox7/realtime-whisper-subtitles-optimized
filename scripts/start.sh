#!/bin/bash
# Real-time Whisper Subtitles - Start Script
# Start the application with various profiles
# Encoding: UTF-8

set -e

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

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --monitoring    Start with monitoring (Prometheus + Grafana)"
    echo "  -p, --production    Start in production mode"
    echo "  -d, --development   Start in development mode"
    echo "  -b, --build         Force rebuild before start"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                  # Start basic application"
    echo "  $0 -m               # Start with monitoring"
    echo "  $0 -p -m            # Start production with monitoring"
    echo "  $0 -d               # Start in development mode"
}

# Default options
MONITORING=false
PRODUCTION=false
DEVELOPMENT=false
BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--monitoring)
            MONITORING=true
            shift
            ;;
        -p|--production)
            PRODUCTION=true
            shift
            ;;
        -d|--development)
            DEVELOPMENT=true
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "? Starting Real-time Whisper Subtitles"
echo "======================================"

# Check if .env exists
if [ ! -f .env ]; then
    log_warning ".env file not found, copying from .env.example"
    cp .env.example .env
fi

# Build if requested
if [ "$BUILD" = true ]; then
    log_info "Building Docker images..."
    docker-compose build --no-cache
    log_success "Build completed"
fi

# Construct docker-compose command
COMPOSE_CMD="docker-compose"
COMPOSE_FILES=""

# Add monitoring profile
if [ "$MONITORING" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --profile monitoring"
    log_info "Monitoring enabled (Prometheus + Grafana)"
fi

# Add production or development overrides
if [ "$PRODUCTION" = true ]; then
    if [ -f "docker-compose.prod.yml" ]; then
        COMPOSE_CMD="$COMPOSE_CMD -f docker-compose.yml -f docker-compose.prod.yml"
        log_info "Production mode enabled"
    else
        log_warning "docker-compose.prod.yml not found, using default configuration"
    fi
elif [ "$DEVELOPMENT" = true ]; then
    if [ -f "docker-compose.dev.yml" ]; then
        COMPOSE_CMD="$COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml"
        log_info "Development mode enabled"
    else
        log_warning "docker-compose.dev.yml not found, using default configuration"
    fi
fi

# Start services
log_info "Starting services..."
eval "$COMPOSE_CMD up -d"

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 10

# Check service health
log_info "Checking service health..."
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    log_success "Application is running and healthy!"
else
    log_warning "Application may still be starting up..."
fi

# Show access URLs
echo
log_success "Services started successfully!"
echo "Access URLs:"
echo "  ? Web Application: http://localhost:8000"

if [ "$MONITORING" = true ]; then
    echo "  ? Prometheus: http://localhost:9090"
    echo "  ? Grafana: http://localhost:3000 (admin/admin)"
fi

echo
log_info "To view logs: docker-compose logs -f"
log_info "To stop: docker-compose down"
echo
