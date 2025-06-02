#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Audio Processing Module
Advanced audio preprocessing and optimization utilities

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
import webrtcvad
from scipy import signal
import noisereduce as nr

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Advanced audio processing for optimal Whisper performance"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        vad_mode: int = 3,
        enable_noise_reduction: bool = True,
        enable_normalization: bool = True
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = webrtcvad.Vad(vad_mode)
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_normalization = enable_normalization
        
        # Audio processing parameters
        self.hop_length = 512
        self.n_fft = 2048
        self.win_length = 2048
        
        logger.info(f"AudioProcessor initialized: SR={sample_rate}, Chunk={chunk_size}")
    
    def preprocess_audio(
        self,
        audio_data: Union[np.ndarray, bytes],
        source_sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Comprehensive audio preprocessing for Whisper
        
        Args:
            audio_data: Raw audio data (numpy array or bytes)
            source_sample_rate: Original sample rate (if known)
            
        Returns:
            Preprocessed audio array normalized for Whisper
        """
        try:
            # Convert bytes to numpy array if necessary
            if isinstance(audio_data, bytes):
                audio_array = self._bytes_to_array(audio_data)
            else:
                audio_array = audio_data.copy()
            
            # Ensure float32 format
            if audio_array.dtype != np.float32:
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    audio_array = audio_array.astype(np.float32)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample if necessary
            if source_sample_rate and source_sample_rate != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=source_sample_rate,
                    target_sr=self.sample_rate,
                    res_type='kaiser_best'
                )
            
            # Apply preprocessing pipeline
            if self.enable_noise_reduction:
                audio_array = self._reduce_noise(audio_array)
            
            if self.enable_normalization:
                audio_array = self._normalize_audio(audio_array)
            
            # Apply bandpass filter for speech frequencies
            audio_array = self._apply_speech_filter(audio_array)
            
            # Remove silence from beginning and end
            audio_array = self._trim_silence(audio_array)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert byte data to numpy array"""
        try:
            # Assume 16-bit PCM
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_array.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Bytes to array conversion error: {e}")
            return np.array([], dtype=np.float32)
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction using spectral gating"""
        try:
            if len(audio) < self.sample_rate:  # Skip if too short
                return audio
                
            # Use noisereduce library for spectral gating
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=0.8
            )
            return reduced_audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        try:
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_rms = 0.2  # Target RMS level
                audio = audio * (target_rms / rms)
            
            # Peak limiting
            max_val = np.max(np.abs(audio))
            if max_val > 0.95:
                audio = audio * (0.95 / max_val)
            
            return audio
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio
    
    def _apply_speech_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter for speech frequencies (80Hz - 8kHz)"""
        try:
            if len(audio) < 100:  # Skip if too short
                return audio
                
            # Design Butterworth bandpass filter
            low_freq = 80.0
            high_freq = min(8000.0, self.sample_rate / 2 - 100)
            
            sos = signal.butter(
                4,
                [low_freq, high_freq],
                btype='band',
                fs=self.sample_rate,
                output='sos'
            )
            
            # Apply filter
            filtered_audio = signal.sosfilt(sos, audio)
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Speech filter failed: {e}")
            return audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end"""
        try:
            if len(audio) < 100:
                return audio
                
            # Find non-silent regions
            non_silent = np.abs(audio) > threshold
            
            if not np.any(non_silent):
                return audio  # All silence, return as-is
            
            # Find first and last non-silent samples
            first_non_silent = np.argmax(non_silent)
            last_non_silent = len(audio) - np.argmax(non_silent[::-1]) - 1
            
            # Add small padding
            padding = int(0.1 * self.sample_rate)  # 100ms padding
            start = max(0, first_non_silent - padding)
            end = min(len(audio), last_non_silent + padding)
            
            return audio[start:end]
            
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def detect_speech_activity(
        self,
        audio_data: bytes,
        frame_duration: int = 30
    ) -> bool:
        """
        Detect speech activity using WebRTC VAD
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            frame_duration: Frame duration in milliseconds (10, 20, or 30)
            
        Returns:
            True if speech is detected
        """
        try:
            if len(audio_data) == 0:
                return False
            
            # Ensure sample rate is supported by WebRTC VAD
            supported_rates = [8000, 16000, 32000, 48000]
            vad_sample_rate = min(supported_rates, key=lambda x: abs(x - self.sample_rate))
            
            # Calculate frame size
            frame_size = int(vad_sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample
            
            if len(audio_data) < frame_size:
                return False
            
            # Use only the required frame size
            frame_data = audio_data[:frame_size]
            
            # Detect speech
            is_speech = self.vad.is_speech(frame_data, vad_sample_rate)
            return is_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to processing if VAD fails
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """Extract audio features for analysis"""
        try:
            features = {}
            
            # Basic statistics
            features['rms'] = np.sqrt(np.mean(audio ** 2))
            features['peak'] = np.max(np.abs(audio))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral features
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
                S=magnitude, sr=self.sample_rate
            ))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
                S=magnitude, sr=self.sample_rate
            ))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
                S=magnitude, sr=self.sample_rate
            ))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def resample_audio(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int,
        method: str = 'kaiser_best'
    ) -> np.ndarray:
        """High-quality audio resampling"""
        try:
            if source_sr == target_sr:
                return audio
            
            resampled = librosa.resample(
                audio,
                orig_sr=source_sr,
                target_sr=target_sr,
                res_type=method
            )
            return resampled
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio
    
    def save_audio(
        self,
        audio: np.ndarray,
        filepath: str,
        sample_rate: Optional[int] = None
    ) -> bool:
        """Save audio to file"""
        try:
            sr = sample_rate or self.sample_rate
            sf.write(filepath, audio, sr)
            logger.info(f"Audio saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False

class StreamingAudioProcessor:
    """Streaming audio processor for real-time applications"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        overlap: float = 0.25
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap)
        self.buffer = np.array([], dtype=np.float32)
        self.processor = AudioProcessor(sample_rate, chunk_size)
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process audio chunk with overlap handling"""
        try:
            # Add chunk to buffer
            self.buffer = np.concatenate([self.buffer, audio_chunk])
            
            # Check if we have enough data
            if len(self.buffer) < self.chunk_size:
                return None
            
            # Extract chunk for processing
            process_chunk = self.buffer[:self.chunk_size]
            
            # Update buffer with overlap
            self.buffer = self.buffer[self.chunk_size - self.overlap_size:]
            
            # Process chunk
            processed = self.processor.preprocess_audio(process_chunk)
            
            return processed if len(processed) > 0 else None
            
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
            return None
    
    def reset(self):
        """Reset the streaming buffer"""
        self.buffer = np.array([], dtype=np.float32)
        logger.debug("Streaming buffer reset")
