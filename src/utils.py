#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Utilities and Helper Functions
Common utilities and helper functions for the application

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import os
import json
import logging
import hashlib
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_data: Dict = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load environment variables"""
        for key, value in os.environ.items():
            config_key = key.lower()
            self.config_data[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            return int(value)
        except ValueError:
            pass
        
        try:
            return float(value)
        except ValueError:
            pass
        
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_data.get(key.lower(), default)

class ValidationUtils:
    """Input validation utilities"""
    
    @staticmethod
    def validate_audio_file(file_path: Union[str, Path]) -> bool:
        """Validate if file is a supported audio format"""
        supported_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
        file_path = Path(file_path)
        return file_path.suffix.lower() in supported_extensions
    
    @staticmethod
    def validate_language_code(language: str) -> bool:
        """Validate language code"""
        supported_languages = {
            'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca',
            'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr',
            'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it',
            'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv',
            'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no',
            'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn',
            'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr',
            'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
        }
        return language.lower() in supported_languages
    
    @staticmethod
    def validate_model_name(model: str) -> bool:
        """Validate Whisper model name"""
        valid_models = {'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'}
        return model in valid_models

class SubtitleExporter:
    """Subtitle export utilities"""
    
    @staticmethod
    def to_srt(segments: List[Dict]) -> str:
        """Export segments to SRT format"""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = SubtitleExporter._seconds_to_srt_time(segment['start'])
            end_time = SubtitleExporter._seconds_to_srt_time(segment['end'])
            text = segment['text'].strip()
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Empty line
        
        return "\n".join(srt_content)
    
    @staticmethod
    def to_vtt(segments: List[Dict]) -> str:
        """Export segments to WebVTT format"""
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = SubtitleExporter._seconds_to_vtt_time(segment['start'])
            end_time = SubtitleExporter._seconds_to_vtt_time(segment['end'])
            text = segment['text'].strip()
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")  # Empty line
        
        return "\n".join(vtt_content)
    
    @staticmethod
    def to_json(segments: List[Dict]) -> str:
        """Export segments to JSON format"""
        return json.dumps(segments, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

# Global utilities instances
_global_config: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration manager"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config
