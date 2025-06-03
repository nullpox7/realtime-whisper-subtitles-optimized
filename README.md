# Real-time Whisper Subtitles - Stream Edition (v2.1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CUDA 12.9.0](https://img.shields.io/badge/CUDA-12.9.0-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Real-time speech recognition and subtitle generation optimized for live streaming**

OpenAI Whisper + CUDA 12.9.0 + cuDNN optimized Web application with streaming focus

## Latest Updates - v2.1.0 (2025-06-03) - Stream Edition

**NEW: OPTIMIZED FOR LIVE STREAMING**

### ? Stream Features Added
- **Microphone Device Selection**: Choose from available audio input devices
- **Fullscreen Subtitle Display**: Black background, large white text for streaming overlays
- **No History Mode**: Live subtitles replace previous text (perfect for OBS/streaming)
- **Keyboard Shortcuts**: F for fullscreen, Space for record toggle, C for clear
- **Streaming UI**: Clean, minimal interface optimized for broadcasters

### ? Technical Improvements
- **File Upload Removed**: Streamlined for real-time use only
- **Energy-based Speech Detection**: Reliable without external dependencies
- **UTF-8 Encoding Fixed**: Complete English UI eliminates encoding issues
- **Simplified Audio Processing**: Direct microphone input with device selection

### Quick Start for Streamers
```bash
# Download and run
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized
docker-compose up -d

# Access at http://localhost:8000
# Press F for fullscreen subtitle overlay
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
- **GPU Acceleration** with Real-Time Factor < 1.0
- **WebSocket-based** real-time transcription
- **Energy-based Speech Detection** (reliable and fast)
- **Optimized Models**: From tiny (fastest) to large-v3 (best quality)

### ? AI Capabilities
- **OpenAI Whisper** models (tiny to large-v3)
- **faster-whisper** optimization
- **Auto Language Detection** + 90+ languages
- **Word-level Timestamps** with confidence scores

### ? Modern Interface
- **Stream-focused UI**: Minimal, clean design
- **Collapsible Statistics**: Hide/show performance metrics
- **Responsive Design**: Works on desktop and mobile
- **Dark Theme Support**: Perfect for streaming setups

## Streaming Setup Guide

### For OBS Studio
1. Start the application: `docker-compose up -d`
2. Open http://localhost:8000
3. Select your microphone device
4. Choose language and model
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

## Quick Start

### Prerequisites
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+ (optional but recommended)
- **VRAM**: 4GB+ (8GB+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **CPU**: 4+ cores
- **Microphone**: Any USB or built-in microphone

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized
```

#### 2. Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Edit configuration if needed (optional)
nano .env
```

#### 3. Create Data Directories
```bash
mkdir -p data/{models,outputs,logs,cache}
chmod -R 755 data/
```

#### 4. Run with Docker Compose
```bash
# Basic startup
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

#### 5. Access Application
Open your browser and navigate to `http://localhost:8000`

## Configuration

### Model Selection (Speed vs Quality)
- **tiny**: Fastest, good for real-time streaming
- **base**: Balanced (recommended for most streams)
- **small**: Better quality, slightly slower
- **medium**: High quality, needs more processing power

### Language Settings
- **auto**: Auto-detect language (recommended)
- **Specific**: Choose if you know the primary language

### Microphone Setup
1. Grant microphone permissions in browser
2. Select your preferred microphone from dropdown
3. Test recording to ensure audio levels are good

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

## Troubleshooting

### Common Issues

#### No Microphone Detected
```bash
# Check browser permissions
# Refresh page after granting microphone access
# Try different browser (Chrome recommended)
```

#### Audio Not Processing
```bash
# Check microphone levels
# Verify WebSocket connection (green status indicator)
# Check container logs: docker-compose logs whisper-subtitles
```

#### Performance Issues
```bash
# Use smaller model (tiny/base)
# Check GPU usage: nvidia-smi
# Reduce audio quality in browser settings
```

### Quick Fix Script
```bash
# Download and run the complete fix script
curl -O https://raw.githubusercontent.com/nullpox7/realtime-whisper-subtitles-optimized/main/quick_fix_complete.sh
chmod +x quick_fix_complete.sh
./quick_fix_complete.sh
```

### Health Check
```bash
# Check application status
curl http://localhost:8000/health

# Should return:
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true/false,
  "version": "2.1.0"
}
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

## Advanced Configuration

### Performance Tuning
```bash
# High quality (slower)
WHISPER_MODEL=large-v3
BEAM_SIZE=5
TEMPERATURE=0.0

# Fast processing (recommended for streaming)
WHISPER_MODEL=base
BEAM_SIZE=1
TEMPERATURE=0.2
```

### GPU Memory Management
```bash
# Memory optimization
CUDA_MEMORY_FRACTION=0.8
BATCH_SIZE=8

# Model optimization
COMPUTE_TYPE=int8  # Use int8 for faster processing
```

## Monitoring & Logging

### Real-time Monitoring
```bash
# With Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000 (admin/admin)
```

### Log Monitoring
```bash
# View application logs
docker-compose logs -f whisper-subtitles

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## What's New in Stream Edition (v2.1.0)

### Streaming Optimizations
- **Microphone Device Selection**: Choose from available audio input devices
- **Fullscreen Subtitle Display**: Perfect for OBS overlay integration
- **No History Mode**: Clean live subtitles without scrolling history
- **Keyboard Shortcuts**: Quick controls for streamers

### Removed Features (Streamlined)
- **File Upload**: Removed to focus on real-time streaming
- **Complex History**: Simplified to single subtitle display
- **Heavy Statistics**: Made collapsible to reduce clutter

### Technical Improvements
- **Direct Audio Processing**: Simplified pipeline for lower latency
- **Better Error Handling**: Graceful fallbacks for stream reliability
- **WebSocket Optimization**: Improved real-time communication
- **Memory Efficiency**: Optimized for long streaming sessions

## Use Cases

### Perfect For:
- **Live Streaming**: Twitch, YouTube, Facebook Gaming
- **Virtual Meetings**: Zoom, Teams, Discord streams
- **Content Creation**: Podcasts, tutorials, presentations
- **Accessibility**: Real-time captions for hearing impaired viewers
- **Language Learning**: Live translation demonstrations
- **Gaming Streams**: Accessible gaming content

### Example Streaming Setups:
1. **Solo Gaming Stream**: Auto-detect language, base model, fullscreen overlay
2. **Multilingual Stream**: Specific language selection, medium model
3. **Podcast Recording**: Higher quality model, statistics visible
4. **Virtual Meeting**: Auto-detect, collapsible stats, clean UI

## System Requirements

### Minimum (CPU-only)
- **CPU**: 4+ cores
- **RAM**: 8GB
- **Model**: tiny or base
- **Expected RTF**: 2-4x (slower than real-time)

### Recommended (GPU)
- **GPU**: NVIDIA GTX 1060 / RTX 2060 or better
- **VRAM**: 4GB+
- **CPU**: 6+ cores
- **RAM**: 16GB
- **Model**: base or small
- **Expected RTF**: 0.3-0.8x (faster than real-time)

### High Performance (GPU)
- **GPU**: NVIDIA RTX 3070 / RTX 4060 or better
- **VRAM**: 8GB+
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Model**: medium or large-v3
- **Expected RTF**: 0.1-0.5x (much faster than real-time)

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/stream-improvement`)
3. Commit your changes (`git commit -m 'Add stream feature'`)
4. Push to the branch (`git push origin feature/stream-improvement`)
5. Open a Pull Request

### Development Setup
```bash
# Clone repository
git clone https://github.com/nullpox7/realtime-whisper-subtitles-optimized.git
cd realtime-whisper-subtitles-optimized

# Development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

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

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community help
- **Discord**: Real-time community support (coming soon)

### Professional Support
For enterprise deployments and custom integrations, contact us through GitHub Issues with the "enterprise" label.

## Roadmap

### v2.2.0 (Planned)
- **Custom Font Selection**: Choose fonts for subtitle display
- **Color Customization**: Custom text and background colors
- **Position Controls**: Adjust subtitle position on screen
- **Transparency**: Adjustable background transparency
- **Hotkey Customization**: Custom keyboard shortcuts

### v2.3.0 (Future)
- **Multiple Language Support**: Real-time language switching
- **Audio Effects**: Noise reduction and audio enhancement
- **Cloud Integration**: Remote model hosting options
- **Mobile App**: Dedicated mobile application

### v3.0.0 (Vision)
- **AI Translation**: Real-time translation between languages
- **Voice Cloning**: AI voice synthesis for accessibility
- **Advanced Analytics**: Detailed speech analysis and insights
- **Plugin System**: Third-party integrations and extensions

---

**Perfect for streamers, content creators, and accessibility-focused applications. Get real-time, accurate subtitles with minimal setup!**