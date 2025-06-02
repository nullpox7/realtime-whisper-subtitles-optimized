#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Whisper Optimization Module
Enhanced Whisper model management and optimization utilities

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import logging
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from faster_whisper import WhisperModel
import psutil

logger = logging.getLogger(__name__)

class OptimizedWhisperModel:
    """Optimized Whisper model with GPU acceleration and caching"""
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 4,
        num_workers: int = 1,
        download_root: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.compute_type = self._determine_compute_type(compute_type)
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        
        # Model state
        self.model: Optional[WhisperModel] = None
        self.model_info: Dict = {}
        self.is_loaded = False
        self.load_time = 0.0
        
        # Performance tracking
        self.transcription_count = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"OptimizedWhisperModel initialized: {model_name} on {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _determine_compute_type(self, compute_type: str) -> str:
        """Determine the best compute type for the device"""
        if compute_type == "auto":
            if self.device == "cuda":
                # Check GPU memory and CUDA capability
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if gpu_memory >= 8:  # 8GB+ VRAM
                        return "float16"
                    else:
                        return "int8"
                return "float16"
            else:
                return "int8"
        return compute_type
    
    def load_model(self) -> bool:
        """Load the Whisper model with optimization"""
        try:
            with self._lock:
                if self.is_loaded:
                    return True
                
                start_time = time.time()
                
                # Set up model parameters
                model_kwargs = {
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "cpu_threads": self.cpu_threads,
                }
                
                if self.download_root:
                    model_kwargs["download_root"] = self.download_root
                
                # Load model
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = WhisperModel(self.model_name, **model_kwargs)
                
                self.load_time = time.time() - start_time
                self.is_loaded = True
                
                # Store model info
                self.model_info = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "load_time": self.load_time,
                    "cpu_threads": self.cpu_threads
                }
                
                logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float]] = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        initial_prompt: Optional[str] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = None,
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'?([{-",
        append_punctuations: str = "\"'.?,?!?:?)]};",
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None
    ) -> Dict:
        """Enhanced transcription with comprehensive options"""
        try:
            if not self.is_loaded:
                if not self.load_model():
                    raise RuntimeError("Model not loaded")
            
            start_time = time.time()
            
            # Prepare VAD parameters
            if vad_parameters is None:
                vad_parameters = {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": 30,
                    "min_silence_duration_ms": 100,
                    "window_size_samples": 1024,
                    "speech_pad_ms": 30
                }
            
            with self._lock:
                # Perform transcription
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    best_of=best_of,
                    patience=patience,
                    length_penalty=length_penalty,
                    temperature=temperature,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    initial_prompt=initial_prompt,
                    prefix=prefix,
                    suppress_blank=suppress_blank,
                    suppress_tokens=suppress_tokens,
                    without_timestamps=without_timestamps,
                    max_initial_timestamp=max_initial_timestamp,
                    word_timestamps=word_timestamps,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters
                )
            
            # Process segments
            processed_segments = []
            for segment in segments:
                segment_dict = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    segment_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                
                processed_segments.append(segment_dict)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self.transcription_count += 1
            self.total_audio_duration += info.duration
            self.total_processing_time += processing_time
            
            # Calculate real-time factor
            rtf = processing_time / info.duration if info.duration > 0 else 0
            
            result = {
                "success": True,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": getattr(info, 'duration_after_vad', info.duration),
                "transcription_options": {
                    "beam_size": beam_size,
                    "best_of": best_of,
                    "temperature": temperature,
                    "word_timestamps": word_timestamps
                },
                "segments": processed_segments,
                "text": " ".join([seg["text"] for seg in processed_segments]),
                "processing_time": processing_time,
                "real_time_factor": rtf,
                "model_info": self.model_info
            }
            
            logger.debug(f"Transcription completed: {processing_time:.2f}s, RTF: {rtf:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": [],
                "processing_time": 0.0,
                "real_time_factor": 0.0
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_rtf = (
            self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0 else 0
        )
        
        return {
            "transcription_count": self.transcription_count,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "average_rtf": avg_rtf,
            "model_info": self.model_info,
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        memory_info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if torch.cuda.is_available() and self.device == "cuda":
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        return memory_info
    
    def unload_model(self):
        """Unload the model to free memory"""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.is_loaded = False
                logger.info("Model unloaded and memory freed")

class WhisperModelManager:
    """Manager for multiple Whisper models with dynamic loading"""
    
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.models: Dict[str, OptimizedWhisperModel] = {}
        self.model_usage: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        logger.info(f"WhisperModelManager initialized (max_models={max_models})")
    
    def get_model(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        **kwargs
    ) -> OptimizedWhisperModel:
        """Get or create a Whisper model"""
        model_key = f"{model_name}_{device}_{compute_type}"
        
        with self._lock:
            # Return existing model if available
            if model_key in self.models:
                self.model_usage[model_key] = time.time()
                return self.models[model_key]
            
            # Check if we need to remove an old model
            if len(self.models) >= self.max_models:
                self._remove_oldest_model()
            
            # Create new model
            model = OptimizedWhisperModel(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                **kwargs
            )
            
            self.models[model_key] = model
            self.model_usage[model_key] = time.time()
            
            logger.info(f"Created new model: {model_key}")
            return model
    
    def _remove_oldest_model(self):
        """Remove the least recently used model"""
        if not self.models:
            return
        
        # Find oldest model
        oldest_key = min(self.model_usage.keys(), key=self.model_usage.get)
        
        # Unload and remove
        self.models[oldest_key].unload_model()
        del self.models[oldest_key]
        del self.model_usage[oldest_key]
        
        logger.info(f"Removed oldest model: {oldest_key}")
    
    def get_model_stats(self) -> Dict:
        """Get statistics for all models"""
        stats = {}
        with self._lock:
            for key, model in self.models.items():
                stats[key] = model.get_performance_stats()
        return stats
    
    def cleanup(self):
        """Cleanup all models"""
        with self._lock:
            for model in self.models.values():
                model.unload_model()
            self.models.clear()
            self.model_usage.clear()
        logger.info("All models cleaned up")

class BatchWhisperProcessor:
    """Batch processor for multiple audio files"""
    
    def __init__(
        self,
        model_manager: WhisperModelManager,
        max_workers: int = 4
    ):
        self.model_manager = model_manager
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"BatchWhisperProcessor initialized (max_workers={max_workers})")
    
    def process_batch(
        self,
        audio_files: List[Tuple[np.ndarray, Dict]],
        model_name: str = "base",
        **transcribe_kwargs
    ) -> List[Dict]:
        """Process multiple audio files in parallel"""
        try:
            # Submit all tasks
            futures = []
            for audio_data, metadata in audio_files:
                future = self.executor.submit(
                    self._process_single_audio,
                    audio_data,
                    metadata,
                    model_name,
                    **transcribe_kwargs
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "text": ""
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    def _process_single_audio(
        self,
        audio_data: np.ndarray,
        metadata: Dict,
        model_name: str,
        **transcribe_kwargs
    ) -> Dict:
        """Process a single audio file"""
        try:
            model = self.model_manager.get_model(model_name)
            result = model.transcribe(audio_data, **transcribe_kwargs)
            result["metadata"] = metadata
            return result
        except Exception as e:
            logger.error(f"Single audio processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": metadata
            }
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)
        logger.info("BatchWhisperProcessor shutdown")

# Global model manager instance
_global_model_manager: Optional[WhisperModelManager] = None

def get_global_model_manager() -> WhisperModelManager:
    """Get or create the global model manager"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = WhisperModelManager()
    return _global_model_manager

def cleanup_global_resources():
    """Cleanup global resources"""
    global _global_model_manager
    if _global_model_manager is not None:
        _global_model_manager.cleanup()
        _global_model_manager = None
    logger.info("Global resources cleaned up")
