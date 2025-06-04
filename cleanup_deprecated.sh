#!/bin/bash
# Real-time Whisper Subtitles - Cleanup Deprecated Files
# Remove legacy debugging and fix scripts that are no longer needed
# Encoding: UTF-8

set -e

echo "? Real-time Whisper Subtitles - Repository Cleanup"
echo "=================================================="
echo ""
echo "This script will remove deprecated files that are no longer needed."
echo "All functionality has been integrated into the main codebase."
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

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    log_error "This directory is not a Git repository"
    exit 1
fi

# List of deprecated files to remove
deprecated_files=(
    # Debug scripts (development phase only)
    "debug_script.py"
    "fix_numba_error.py"
    
    # Emergency fix scripts (issues resolved)
    "emergency_fix_webrtc.sh"
    "fix_cuda_image.sh"
    "fix_docker_build.sh"
    "fix_gpu_docker_build.sh"
    "diagnose_docker_build.sh"
    "quick_fix_complete.sh"
    "quick_fix_issues.sh"
    "quick_fix_numba.sh"
    "quick_fix_webrtc_vad.sh"
    
    # Duplicate/redundant Dockerfiles
    "Dockerfile.fixed"
    "Dockerfile.emergency"
    "Dockerfile.gpu.fixed"
    "Dockerfile.gpu.lite"
    "Dockerfile.gpu.stable"
    "Dockerfile.gpu.working"
    
    # Redundant docker-compose files
    "docker-compose.emergency.yml"
    "docker-compose.gpu.working.yml"
    
    # Redundant requirements files
    "requirements_fixed.txt"
    "requirements.gpu.working.txt"
    
    # Outdated documentation
    "FIX_SUMMARY.md"
    "GPU_BUILD_FIX_GUIDE.md"
)

# Show what will be removed
echo "? Files to be removed:"
echo ""
existing_files=()
for file in "${deprecated_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ??  $file"
        existing_files+=("$file")
    else
        echo "  ??  $file (not found)"
    fi
done

echo ""
echo "? Summary:"
echo "  - Total deprecated files: ${#deprecated_files[@]}"
echo "  - Files found to remove: ${#existing_files[@]}"
echo "  - Files already removed: $((${#deprecated_files[@]} - ${#existing_files[@]}))"
echo ""

if [ ${#existing_files[@]} -eq 0 ]; then
    log_success "? All deprecated files have already been removed!"
    exit 0
fi

# Confirm removal
read -p "? Do you want to remove these ${#existing_files[@]} deprecated files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "? Cleanup cancelled by user"
    exit 0
fi

echo ""
log_info "?? Removing deprecated files..."

# Remove files using Git
removed_count=0
for file in "${existing_files[@]}"; do
    if git rm "$file" >/dev/null 2>&1; then
        log_success "Removed: $file"
        ((removed_count++))
    else
        log_warning "Failed to remove: $file (may already be staged for removal)"
    fi
done

echo ""
log_success "? Removed $removed_count deprecated files"

# Commit the changes
echo ""
read -p "? Do you want to commit these changes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "? Committing changes..."
    
    if git commit -m "Clean up repository: Remove $removed_count deprecated files

Removed legacy debugging and emergency fix scripts:
- Debug scripts from development phase
- Emergency fix scripts (issues resolved)  
- Duplicate/redundant Dockerfiles
- Redundant docker-compose files
- Redundant requirements files
- Outdated documentation

All functionality has been integrated into main codebase.
Repository is now cleaner and easier to maintain."; then
        log_success "? Changes committed successfully"
        
        echo ""
        read -p "? Do you want to push changes to remote? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "? Pushing to remote repository..."
            if git push; then
                log_success "? Changes pushed to remote repository"
            else
                log_warning "?? Failed to push to remote. You may need to push manually."
            fi
        fi
    else
        log_warning "?? Failed to commit changes. You may need to commit manually."
    fi
else
    log_info "?? Changes staged but not committed. You can commit manually with:"
    echo "   git commit -m 'Remove deprecated files'"
fi

echo ""
log_success "? Repository cleanup completed!"
echo ""
echo "? Remaining essential files:"
echo "  ? Dockerfile, Dockerfile.gpu, Dockerfile.cpu"
echo "  ? docker-compose.yml, docker-compose.gpu.yml, docker-compose.cpu.yml"
echo "  ? requirements.txt, requirements.gpu.txt"  
echo "  ? setup.sh, setup_gpu.sh, quick_start.sh"
echo "  ? scripts/ directory (cleanup.sh, start.sh, etc.)"
echo "  ? src/, static/, templates/, config/ directories"
echo "  ? README.md, README_GPU.md, SETUP.md"
echo ""
echo "? Your repository is now clean and ready for production use!"
echo ""
echo "? Next steps:"
echo "  - Run: ./setup.sh (standard setup)"  
echo "  - Run: ./setup_gpu.sh (GPU setup)"
echo "  - Run: ./quick_start.sh (universal setup)"
echo "  - Check: CLEANUP.md for more information"
