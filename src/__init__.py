#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Main Package
High-performance real-time speech recognition and subtitle generation

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

__version__ = "2.0.0"
__author__ = "Real-time Whisper Subtitles Team"
__license__ = "MIT"
__description__ = "Real-time speech recognition and subtitle generation using OpenAI Whisper with CUDA optimization"

# Package imports
from .web_interface import app
from .audio_processing import AudioProcessor, StreamingAudioProcessor
from .whisper_optimizer import OptimizedWhisperModel, WhisperModelManager
from .monitoring import WhisperMetrics, SystemMonitor, get_metrics
from .utils import ConfigManager, ValidationUtils, SubtitleExporter, get_config

__all__ = [
    "app",
    "AudioProcessor",
    "StreamingAudioProcessor", 
    "OptimizedWhisperModel",
    "WhisperModelManager",
    "WhisperMetrics",
    "SystemMonitor",
    "ConfigManager",
    "ValidationUtils",
    "SubtitleExporter",
    "get_metrics",
    "get_config"
]
