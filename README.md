# Real-time Whisper Subtitles - Stream Edition (v2.2.1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Whisper Large-v3](https://img.shields.io/badge/Whisper-Large--v3-blue.svg)](https://github.com/openai/whisper)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Real-time speech recognition and subtitle generation optimized for live streaming**

OpenAI Whisper + CUDA 12.9 + Large-v3 model optimized Web application with streaming focus

## ? GPU Edition Available - [README_GPU.md](README_GPU.md)

**NEW: CUDA 12.9 + Large-v3 Model Support**
- **Maximum Accuracy**: 97%+ accuracy with Large-v3 model
- **CUDA 12.9 Optimized**: Latest GPU acceleration technology
- **Real-time Factor < 0.3x**: Process audio 3x faster than real-time
- **Enterprise Ready**: Production-grade GPU deployment

[? **GPU Setup Guide**](README_GPU.md) | [? **Quick GPU Setup**](setup_gpu.sh) | [? **Repository Cleanup**](CLEANUP.md)

## Latest Updates - v2.2.1 (2025-06-04) - Repository Organized

**NEW: REPOSITORY CLEANUP + SETUP IMPROVEMENTS**

### ? Repository Organization
- **Clean Structure**: Removed legacy debugging and fix scripts
- **Setup Script Added**: `./setup.sh` now available in root directory
- **File Management**: Clear separation of essential vs deprecated files
- **Easy Cleanup**: Use `./cleanup_deprecated.sh` to remove old files

### ? Streamlined Setup Options
- **Universal Setup**: `./setup.sh` - Standard installation
- **GPU Setup**: `./setup_gpu.sh` - GPU-optimized installation
- **Quick Start**: `./quick_start.sh` - Auto-detects and configures system
- **Repository Cleanup**: `./cleanup_deprecated.sh` - Removes unnecessary files

### ? GPU Edition Features
- **CUDA 12.9 Optimization**: Latest NVIDIA GPU acceleration
- **Large-v3 Model**: Maximum accuracy speech recognition (97%+)
- **Real-time Factor < 0.3x**: Process audio 3x faster than real-time
- **Memory Optimized**: Smart GPU memory management
- **Multi-GPU Support**: Scale across multiple GPUs

### ? Stream Features (Continued)
- **Microphone Device Selection**: Choose from available audio input devices
- **Fullscreen Subtitle Display**: Black background, large white text for streaming overlays
- **No History Mode**: Live subtitles replace previous text (perfect for OBS/streaming)
- **Keyboard Shortcuts**: F for fullscreen, Space for record toggle, C for clear
- **Streaming UI**: Clean, minimal interface optimized for broadcasters

### ? Technical Improvements
- **Energy-based Speech Detection**: Reliable without external dependencies
- **UTF-8 Encoding Fixed**: Complete English UI eliminates encoding issues
- **Simplified Audio Processing**: Direct microphone input with device selection
- **Clean Repository Structure**: Essential files only, deprecated files removable

## Quick Start Options

### ? Universal Quick Start (Recommended)
```bash
# Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# Auto-setup (detects GPU/CPU and configures automatically)
chmod +x quick_start.sh
./quick_start.sh

# Access at http://localhost:8000
```

### ? GPU Edition (Maximum Performance)
```bash
# Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# GPU setup (automated)
chmod +x setup_gpu.sh
./setup_gpu.sh

# Access at http://localhost:8000
```

### ?? Standard Edition (CPU/Basic GPU)
```bash
# Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# Standard setup
chmod +x setup.sh
./setup.sh

# Or manual setup
cp .env.example .env
docker-compose up -d

# Access at http://localhost:8000
```

### ? Repository Cleanup (Optional)
```bash
# Remove deprecated debugging and fix scripts
chmod +x cleanup_deprecated.sh
./cleanup_deprecated.sh

# This removes legacy files that are no longer needed
# See CLEANUP.md for detailed information
```

---

## Key Features

### ? Streaming Optimized
- **Real-time Subtitles**: Live speech-to-text with minimal delay
- **Microphone Selection**: Choose specific audio input device
- **Fullscreen Mode**: Black background overlay perfect for OBS/streaming software
- **No History**: Clean display showing only current speech (no scrolling)
- **Keyboard Controls**: Space to start/stop, F for fullscreen, C to clear

### ? Performance
- **GPU Acceleration** with Real-Time Factor < 0.3x (GPU Edition)
- **WebSocket-based** real-time transcription
- **Energy-based Speech Detection** (reliable and fast)
- **Optimized Models**: From tiny (fastest) to large-v3 (best quality)

### ? AI Capabilities
- **OpenAI Whisper** models (tiny to large-v3)
- **faster-whisper** optimization
- **Auto Language Detection** + 99+ languages
- **Word-level Timestamps** with confidence scores

### ? Modern Interface
- **Stream-focused UI**: Minimal, clean design
- **Collapsible Statistics**: Hide/show performance metrics
- **Responsive Design**: Works on desktop and mobile
- **Dark Theme Support**: Perfect for streaming setups

## Model Comparison

| Model | Size | VRAM | Speed | Accuracy | Best For |
|-------|------|------|-------|----------|----------|
| tiny | 39MB | 1GB | 32x | 85% | Development/Testing |
| base | 74MB | 1GB | 16x | 89% | General Streaming |
| small | 244MB | 2GB | 6x | 92% | Quality Streaming |
| medium | 769MB | 3GB | 2x | 95% | High Quality |
| **large-v3** | **1.55GB** | **4GB+** | **1x** | **97%** | **Maximum Accuracy** |

## System Requirements

### GPU Edition (Recommended)
- **GPU**: NVIDIA RTX 3060 / RTX 4060 or better
- **VRAM**: 6GB+ (8GB+ recommended for large-v3)
- **CUDA**: 12.1+ with compatible drivers (auto-detected)
- **RAM**: 16GB+ (32GB recommended)
- **CPU**: 6+ cores
- **Expected RTF**: 0.15-0.45x (faster than real-time)

### Standard Edition
- **GPU**: NVIDIA GTX 1060 / RTX 2060 or better (optional)
- **VRAM**: 4GB+ (if using GPU)
- **RAM**: 8GB+ (16GB recommended)
- **CPU**: 4+ cores
- **Expected RTF**: 0.3-2.0x depending on hardware

### CPU-only
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB+ (16GB recommended)
- **Model**: tiny, base, or small recommended
- **Expected RTF**: 2-8x (slower than real-time)

## Setup Scripts

### Available Setup Options

| Script | Purpose | Best For |
|--------|---------|----------|
| `./quick_start.sh` | Universal auto-setup | New users, auto-detection |
| `./setup.sh` | Standard installation | CPU/basic GPU systems |
| `./setup_gpu.sh` | GPU-optimized setup | High-performance GPU systems |
| `./cleanup_deprecated.sh` | Remove old files | Repository maintenance |

### Setup Script Features
- **Auto-detection**: Automatically detects GPU availability and CUDA compatibility
- **Dependency Check**: Verifies Docker, Docker Compose, and NVIDIA drivers
- **Environment Setup**: Creates appropriate `.env` configuration
- **Model Download**: Optional pre-download of Whisper models
- **Health Verification**: Tests application startup and connectivity

## Installation

### Method 1: Universal Quick Start (Easiest)
```bash
# 1. Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# 2. Run universal setup (auto-detects system capabilities)
chmod +x quick_start.sh
./quick_start.sh

# 3. Access application
open http://localhost:8000
```

### Method 2: GPU Edition Setup (Maximum Performance)
```bash
# 1. Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# 2. Check GPU compatibility
nvidia-smi  # Should show CUDA support

# 3. Automated GPU setup
chmod +x setup_gpu.sh
./setup_gpu.sh

# 4. Access application
open http://localhost:8000
```

### Method 3: Standard Edition Setup
```bash
# 1. Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# 2. Standard setup
chmod +x setup.sh
./setup.sh

# 3. Access application
open http://localhost:8000
```

### Method 4: Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# 2. Environment setup
cp .env.example .env

# 3. Create data directories
mkdir -p data/{models,outputs,logs,cache}
chmod -R 755 data/

# 4. Run with Docker Compose
docker-compose up -d

# 5. Access application
open http://localhost:8000
```

## Repository Management

### ? Cleanup Deprecated Files
This repository includes legacy debugging and fix scripts that are no longer needed. You can safely remove them:

```bash
# View what will be removed
./cleanup_deprecated.sh

# The script removes:
# - Legacy debugging scripts (debug_*.py, fix_*.sh)
# - Emergency fix scripts (quick_fix_*.sh)
# - Duplicate Dockerfiles and configurations
# - Outdated documentation files

# See CLEANUP.md for detailed information
```

### ? Essential Files (Keep These)
- **Main Configuration**: `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- **GPU Configuration**: `Dockerfile.gpu`, `docker-compose.gpu.yml`, `requirements.gpu.txt`
- **Setup Scripts**: `setup.sh`, `setup_gpu.sh`, `quick_start.sh`
- **Application Code**: `src/`, `static/`, `templates/`, `config/`
- **Documentation**: `README.md`, `README_GPU.md`, `SETUP.md`

## Troubleshooting

### Quick Health Check
```bash
# Check application status
curl http://localhost:8000/health

# Should return:
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true/false,
  "version": "2.2.1"
}
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| ? Setup script not found | Ensure you're in project root, use `chmod +x setup.sh` |
| ? GPU not detected | Check `nvidia-smi` and NVIDIA Container Toolkit |
| ? Model loading slow | Use smaller model (base/small) or check VRAM |
| ? High latency | Reduce batch size or use faster model |
| ? Audio not working | Check microphone permissions in browser |
| ? Too many files | Run `./cleanup_deprecated.sh` to remove old files |

### Available Docker Configurations

| Configuration | File | Best For |
|---------------|------|----------|
| Standard | `docker-compose.yml` | General use, auto GPU/CPU detection |
| GPU Optimized | `docker-compose.gpu.yml` | High-performance GPU systems |
| CPU Only | `docker-compose.cpu.yml` | CPU-only systems |
| Production | `docker-compose.prod.yml` | Production deployment |

## Configuration

### Model Selection Guide
- **tiny**: Fastest, good for low-end hardware or testing
- **base**: Balanced (recommended for standard streaming)
- **small**: Better quality, moderate speed
- **medium**: High quality, needs more processing power  
- **large-v3**: Maximum accuracy, requires GPU for real-time use

### Language Settings
- **auto**: Auto-detect language (recommended)
- **Specific**: Choose if you know the primary language

### GPU Configuration (GPU Edition)
```env
# Maximum accuracy configuration
WHISPER_MODEL=large-v3
LANGUAGE=auto
DEVICE=cuda
COMPUTE_TYPE=float16

# Performance optimization
BEAM_SIZE=5
BEST_OF=5
TEMPERATURE=0.0
CUDA_MEMORY_FRACTION=0.85
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Supported Languages
```bash
curl http://localhost:8000/api/languages
```

### Get Available Models
```bash
curl http://localhost:8000/api/models
```

## Streaming Setup Guide

### For OBS Studio
1. Start the application: `./quick_start.sh` or `docker-compose up -d`
2. Open http://localhost:8000
3. Select your microphone device
4. Choose language and model (large-v3 for best quality)
5. Click fullscreen button or press F
6. In OBS: Add Browser Source with URL: `http://localhost:8000` (fullscreen mode)
7. Set source to fullscreen, enable CSS: `body { margin: 0; }`

### For Other Streaming Software
- **Fullscreen URL**: http://localhost:8000 (press F for fullscreen)
- **Resolution**: Any (responsive)
- **Background**: Black (#000000)
- **Text Color**: White (#FFFFFF) with shadow
- **Refresh Rate**: Real-time via WebSocket

### Keyboard Shortcuts
- **F**: Toggle fullscreen subtitle display
- **Space**: Start/stop recording
- **C**: Clear current subtitle
- **Escape**: Exit fullscreen mode

## Stream Integration Examples

### OBS Studio Integration
```
1. Add Browser Source
2. URL: http://localhost:8000
3. Width: 1920, Height: 1080 (or your stream resolution)
4. Custom CSS (optional):
   body { margin: 0; background: transparent; }
   .fullscreen-subtitle { background: rgba(0,0,0,0.8); }
5. Press F in browser for fullscreen subtitle mode
```

### Streamlabs OBS
```
1. Add Browser Source
2. URL: http://localhost:8000
3. Enable fullscreen mode (press F)
4. Position and resize as needed
```

### XSplit
```
1. Add Web Page source
2. URL: http://localhost:8000
3. Use fullscreen mode for clean overlay
```

## Use Cases

### Perfect For:
- **Live Streaming**: Twitch, YouTube, Facebook Gaming
- **Virtual Meetings**: Zoom, Teams, Discord streams
- **Content Creation**: Podcasts, tutorials, presentations
- **Accessibility**: Real-time captions for hearing impaired viewers
- **Language Learning**: Live translation demonstrations
- **Gaming Streams**: Accessible gaming content

### Example Streaming Setups:
1. **High-end Gaming Stream**: GPU Edition + large-v3 model + auto-detect
2. **Standard Gaming Stream**: Standard Edition + base model + specific language
3. **Podcast Recording**: GPU Edition + large-v3 + visible statistics
4. **Virtual Meeting**: Standard Edition + small model + collapsible stats

## Documentation

- **GPU Setup Guide**: [README_GPU.md](README_GPU.md) - Complete CUDA 12.9 + Large-v3 guide
- **Standard Setup Guide**: [SETUP.md](SETUP.md) - Detailed setup instructions
- **Repository Cleanup**: [CLEANUP.md](CLEANUP.md) - File organization and cleanup guide
- **Issue Tracker**: [GitHub Issues](https://github.com/nullpox7/realtime-whisper-subtitles-optimized/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nullpox7/realtime-whisper-subtitles-optimized/discussions)

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Core speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Performance optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - GPU containerization

## Support

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community help

### Professional Support
For enterprise deployments and custom integrations, contact us through GitHub Issues with the "enterprise" label.

## Roadmap

### v2.3.0 (Planned)
- **Custom Font Selection**: Choose fonts for subtitle display
- **Color Customization**: Custom text and background colors
- **Position Controls**: Adjust subtitle position on screen
- **Transparency**: Adjustable background transparency
- **Advanced GPU Features**: Multi-GPU load balancing

### v2.4.0 (Future)
- **Multiple Language Support**: Real-time language switching
- **Audio Effects**: Advanced noise reduction and audio enhancement
- **Cloud Integration**: Remote model hosting options
- **Mobile App**: Dedicated mobile application

### v3.0.0 (Vision)
- **AI Translation**: Real-time translation between languages
- **Voice Cloning**: AI voice synthesis for accessibility
- **Advanced Analytics**: Detailed speech analysis and insights
- **Plugin System**: Third-party integrations and extensions

---

**Perfect for streamers, content creators, and accessibility-focused applications. Get real-time, accurate subtitles with minimal setup!**

**? Want maximum accuracy? Try the [GPU Edition](README_GPU.md) with CUDA 12.9 + Large-v3 model!**

**? Clean repository? Use our [Cleanup Guide](CLEANUP.md) to remove unnecessary files!**
