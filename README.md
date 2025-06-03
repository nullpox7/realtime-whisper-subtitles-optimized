# Real-time Whisper Subtitles (Fixed v2.0.3)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CUDA 12.9.0](https://img.shields.io/badge/CUDA-12.9.0-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Real-time speech recognition and subtitle generation using OpenAI Whisper with CUDA optimization**

OpenAI Whisper + CUDA 12.9.0 + cuDNN optimized Web application

## Latest Updates - v2.0.3 (2025-06-03) - Critical Fixes

**ALL ISSUES FIXED**

### Issues Resolved
- **Character Encoding Fixed**: Complete English UI eliminates all encoding issues
- **AudioSegment Conversion Failed**: PyDub dependency completely removed
- **Audio Processing Errors**: Simplified processing with librosa/soundfile only
- **WebRTC VAD Issues**: Replaced with reliable energy-based speech detection
- **UTF-8 JSON Responses**: Proper UTF-8 encoding throughout the application

### Quick Fix (Recommended)
```bash
# Download and run the complete fix script
curl -O https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/quick_fix_complete.sh
chmod +x quick_fix_complete.sh
./quick_fix_complete.sh
```

### Manual Update
```bash
# Pull latest fixes
git pull origin main

# Clean rebuild (removes PyDub and problematic dependencies)
docker-compose down
docker system prune -af
docker-compose build --no-cache
docker-compose up -d
```

**Test the fix**: `curl http://localhost:8000/health`

---

## Key Features

### Real-time Processing
- **WebSocket-based** real-time transcription
- **Energy-based Speech Detection** (reliable and fast)
- **GPU Acceleration** with Real-Time Factor < 1.0

### AI Capabilities
- **OpenAI Whisper** models (tiny to large-v3)
- **faster-whisper** optimization
- **Auto Language Detection** + 90+ languages
- **Word-level Timestamps** with confidence scores

### Modern Web Interface
- **Full English UI** (encoding issues fixed)
- **Bootstrap 5 + Font Awesome** responsive design
- **Real-time Statistics** display
- **File Upload Support** for batch processing

### Production Ready
- **Docker containerized** deployment
- **Health monitoring** endpoints
- **Comprehensive logging** with UTF-8 support
- **Error recovery** and graceful degradation

## Quick Start

### Prerequisites
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+ (optional but recommended)
- **VRAM**: 4GB+ (8GB+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **CPU**: 4+ cores
- **Storage**: 10GB+ free space

### System Requirements
- **OS**: Ubuntu 20.04/22.04, CentOS 8+, or Windows with WSL2
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: nvidia-docker2 (for GPU acceleration)

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized
```

### 2. Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Edit configuration if needed
nano .env
```

### 3. Create Data Directories
```bash
mkdir -p data/{models,outputs,logs,cache}
chmod -R 755 data/
```

### 4. Run with Docker Compose
```bash
# Basic startup
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### 5. Access Application
Open your browser and navigate to `http://localhost:8000`

## Troubleshooting

### Quick Fix for All Issues

#### One-command Fix
```bash
# Download and run the complete fix script
./quick_fix_complete.sh
```

#### Manual Troubleshooting

##### Fixed Issue: "AudioSegment conversion failed"
```bash
# COMPLETELY FIXED in v2.0.3
# PyDub dependency has been removed
# Audio processing now uses librosa/soundfile only

# Verification:
curl http://localhost:8000/health
# Should return: "status": "healthy"
```

##### Fixed Issue: Character encoding/display problems
```bash
# COMPLETELY FIXED in v2.0.3
# All text is now in English
# UTF-8 encoding properly configured throughout

# Verification:
curl http://localhost:8000/api/languages
# Should return proper JSON with English language names
```

### NVIDIA Docker Setup

#### Ubuntu/Debian
```bash
# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

## Configuration

### Performance Optimization
```bash
# High quality (slower)
WHISPER_MODEL=large-v3
BEAM_SIZE=5
TEMPERATURE=0.0
ENABLE_WORD_TIMESTAMPS=true

# Fast processing (lower quality)
WHISPER_MODEL=tiny
BEAM_SIZE=1
TEMPERATURE=0.2
ENABLE_WORD_TIMESTAMPS=false
```

### GPU Memory Management
```bash
# Memory optimization
CUDA_MEMORY_FRACTION=0.8
BATCH_SIZE=8  # Reduce if out of memory

# Model optimization
ENABLE_QUANTIZATION=true
COMPUTE_TYPE=int8
```

### Language Configuration
```bash
# Auto-detect (default)
LANGUAGE=auto

# Specific language
LANGUAGE=en  # English
LANGUAGE=ja  # Japanese
LANGUAGE=zh  # Chinese
# ... supports 90+ languages
```

## Monitoring

### Prometheus + Grafana
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access URLs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Health Monitoring
```bash
# Check application health
curl http://localhost:8000/health

# View logs
docker-compose logs -f whisper-subtitles

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## What's Fixed in v2.0.3

### Removed Problematic Dependencies
- **PyDub**: Completely removed to fix "AudioSegment conversion failed"
- **WebRTC VAD**: Replaced with simple energy-based detection
- **noisereduce**: Removed to avoid NumPy compatibility issues

### Improved Audio Processing
- **Direct byte-to-array conversion**: No more PyDub conversion errors
- **librosa/soundfile only**: Stable, well-tested audio libraries
- **Energy-based speech detection**: Simple and reliable
- **Better error handling**: Graceful fallbacks for all audio operations

### Complete English Interface
- **Full English UI**: No more character encoding issues
- **UTF-8 JSON responses**: Proper encoding throughout
- **English error messages**: Clear, readable error reporting
- **Auto-detect language**: Still supports all languages for transcription

### Enhanced Reliability
- **Simplified dependencies**: Fewer moving parts, more stability
- **Better logging**: UTF-8 compatible logging with English messages
- **Robust error handling**: Application continues running despite errors
- **Health monitoring**: Better status reporting and diagnostics

## Usage Examples

### Real-time Transcription
1. Open http://localhost:8000
2. Click "Start Recording"
3. Allow microphone access
4. Speak into your microphone
5. See real-time transcription appear

### File Upload
1. Click on the file upload area
2. Select an audio file (WAV, MP3, M4A, FLAC, OGG)
3. Wait for processing
4. View transcription results

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Get supported languages
curl http://localhost:8000/api/languages

# Get available models
curl http://localhost:8000/api/models

# File transcription
curl -X POST \
  -F "audio_file=@your_audio.wav" \
  -F "language=auto" \
  -F "model=base" \
  http://localhost:8000/api/transcribe
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Core speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Performance optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - GPU containerization

## Documentation

- **Quick Start Guide**: [SETUP.md](SETUP.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/nullpox7/realtime-whisper-subtitles-optimized/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nullpox7/realtime-whisper-subtitles-optimized/discussions)

## Support

If you encounter any issues after applying the fixes:

1. **Run the quick fix script**: `./quick_fix_complete.sh`
2. **Check the logs**: `docker-compose logs whisper-subtitles`
3. **Verify health**: `curl http://localhost:8000/health`
4. **Open an issue**: [GitHub Issues](https://github.com/nullpox7/realtime-whisper-subtitles-optimized/issues)

---

**All major issues have been resolved in v2.0.3. Enjoy stable, real-time speech recognition!**
