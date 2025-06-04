# Repository Cleanup Information

## ? File Organization

This repository has been organized to focus on the essential files for production use. Below is information about file structure and cleanup.

### ?? Deprecated Files (Safe to Remove)

The following files are legacy debugging and emergency fix scripts that are no longer needed:

#### Debug Scripts (Development Phase Only)
- `debug_container.py` - Container debugging utility
- `debug_script.py` - General debugging script  
- `fix_numba_error.py` - Numba error fix script

#### Emergency Fix Scripts (Issues Resolved)
- `emergency_fix_webrtc.sh` - WebRTC VAD fix (resolved)
- `fix_cuda_image.sh` - CUDA image compatibility fix (resolved)
- `fix_docker_build.sh` - Docker build error fix (resolved)
- `fix_gpu_docker_build.sh` - GPU Docker build fix (resolved)
- `diagnose_docker_build.sh` - Build diagnosis tool (resolved)
- `quick_fix_*.sh` - Various quick fix scripts (resolved)

#### Duplicate/Redundant Dockerfiles
- `Dockerfile.fixed` - Duplicate of main Dockerfile
- `Dockerfile.emergency` - Emergency fallback (not needed)
- `Dockerfile.gpu.fixed` - Duplicate of Dockerfile.gpu
- `Dockerfile.gpu.lite` - Lightweight variant (redundant)
- `Dockerfile.gpu.stable` - Stable variant (redundant)
- `Dockerfile.gpu.working` - Working variant (redundant)

#### Redundant Docker Compose Files
- `docker-compose.emergency.yml` - Emergency configuration
- `docker-compose.gpu.working.yml` - Working GPU variant

#### Redundant Requirements Files
- `requirements_fixed.txt` - Duplicate of requirements.txt
- `requirements.gpu.working.txt` - Duplicate of requirements.gpu.txt

#### Outdated Documentation
- `FIX_SUMMARY.md` - Fix summary (issues resolved)
- `GPU_BUILD_FIX_GUIDE.md` - GPU build guide (integrated into README_GPU.md)

### ? Active Files (Keep)

#### Main Configuration Files
- `Dockerfile` - Main Docker configuration
- `Dockerfile.gpu` - GPU-optimized configuration  
- `Dockerfile.cpu` - CPU-only configuration
- `docker-compose.yml` - Main compose configuration
- `docker-compose.gpu.yml` - GPU compose configuration
- `docker-compose.cpu.yml` - CPU compose configuration
- `docker-compose.prod.yml` - Production overrides

#### Requirements Files
- `requirements.txt` - Main Python dependencies
- `requirements.gpu.txt` - GPU-optimized dependencies

#### Setup Scripts
- `setup.sh` - Main setup script (links to scripts/setup.sh)
- `setup_gpu.sh` - GPU-specific setup
- `quick_start.sh` - Universal quick start
- `scripts/setup.sh` - Detailed setup script
- `scripts/cleanup.sh` - Cleanup utility
- `scripts/start.sh` - Service startup script

#### Core Application Files
- `src/` - Application source code
- `static/` - Web interface assets
- `templates/` - HTML templates
- `config/` - Configuration files

#### Documentation
- `README.md` - Main documentation
- `README_GPU.md` - GPU-specific guide
- `SETUP.md` - Setup instructions
- `LICENSE` - MIT license

### ? Cleanup Instructions

To remove deprecated files from your local copy:

```bash
# Download and run cleanup script
curl -O https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/cleanup_deprecated.sh
chmod +x cleanup_deprecated.sh
./cleanup_deprecated.sh
```

Or manually remove using Git:

```bash
git rm debug_container.py debug_script.py fix_numba_error.py
git rm emergency_fix_webrtc.sh fix_cuda_image.sh fix_docker_build.sh
git rm fix_gpu_docker_build.sh diagnose_docker_build.sh
git rm quick_fix_*.sh
git rm Dockerfile.fixed Dockerfile.emergency Dockerfile.gpu.fixed
git rm Dockerfile.gpu.lite Dockerfile.gpu.stable Dockerfile.gpu.working
git rm docker-compose.emergency.yml docker-compose.gpu.working.yml
git rm requirements_fixed.txt requirements.gpu.working.txt
git rm FIX_SUMMARY.md GPU_BUILD_FIX_GUIDE.md
git commit -m "Remove deprecated files"
```

### ? Quick Start

For new users, use these essential commands:

```bash
# Standard setup
./setup.sh

# GPU setup  
./setup_gpu.sh

# Universal quick start (auto-detects system)
./quick_start.sh

# Start services
docker-compose up -d

# GPU services
docker-compose -f docker-compose.gpu.yml up -d
```

---

This organization makes the repository cleaner and easier to maintain while preserving all essential functionality.
