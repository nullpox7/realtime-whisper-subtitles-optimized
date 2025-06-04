#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Main Web Interface (Double Free Fixed v2.2.2)
CUDA 12.4 + cuDNN optimized version - Memory safety enhanced

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set up safe memory management before any heavy imports
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/dev/null')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

# Safe imports with error handling
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

# Core ML/Audio libraries with memory safety
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
except:
    pass

import torchaudio
import numpy as np
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

# Log directory configuration
log_dir = os.getenv('LOG_PATH', '/app/data/logs')
log_file_path = os.path.join(log_dir, 'whisper_app.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration with English defaults"""
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    MODEL_PATH = Path("/app/data/models")
    OUTPUT_PATH = Path("/app/data/outputs")
    STATIC_PATH = Path("/app/static")
    TEMPLATE_PATH = Path("/app/templates")

app = FastAPI(
    title="Real-time Whisper Subtitles",
    description="Real-time speech recognition with faster-whisper (Memory Safe)",
    version="2.2.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Config.STATIC_PATH)), name="static")
templates = Jinja2Templates(directory=str(Config.TEMPLATE_PATH))

class UTF8JSONResponse(JSONResponse):
    """JSON response with proper UTF-8 encoding"""
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

whisper_model: Optional[WhisperModel] = None

class AudioProcessor:
    """Memory-safe audio processor"""
    
    @staticmethod
    def preprocess_audio(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> np.ndarray:
        """Process audio data with memory safety"""
        try:
            if len(audio_data) == 0:
                return np.array([], dtype=np.float32)
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0
            
            if len(audio_array) == 0:
                return np.array([], dtype=np.float32)
            
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            return audio_array
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def detect_speech(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> bool:
        """Energy-based speech detection"""
        try:
            if len(audio_data) == 0:
                return False
            
            audio_array = AudioProcessor.preprocess_audio(audio_data, sample_rate)
            if len(audio_array) == 0:
                return False
            
            energy = np.mean(audio_array ** 2)
            return energy > 0.001
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return True

class WhisperManager:
    """Memory-safe Whisper model management"""
    
    @staticmethod
    def load_model(model_name: str = Config.WHISPER_MODEL) -> WhisperModel:
        """Load Whisper model with memory safety"""
        try:
            model_path = Config.MODEL_PATH / "whisper" / model_name
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if model_path.exists():
                model = WhisperModel(
                    str(model_path),
                    device=Config.DEVICE,
                    compute_type=Config.COMPUTE_TYPE,
                    cpu_threads=4,
                    num_workers=1
                )
            else:
                Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
                model = WhisperModel(
                    model_name,
                    device=Config.DEVICE,
                    compute_type=Config.COMPUTE_TYPE,
                    download_root=str(Config.MODEL_PATH / "whisper"),
                    cpu_threads=4,
                    num_workers=1
                )
            
            logger.info(f"Model loaded: {model_name} on {Config.DEVICE}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    @staticmethod
    async def transcribe(audio_array: np.ndarray, language: str = "auto") -> Dict:
        """Transcribe audio with memory safety"""
        try:
            if whisper_model is None:
                raise ValueError("Whisper model not loaded")
            
            if len(audio_array) == 0:
                return {"success": False, "error": "Empty audio data", "text": ""}
            
            start_time = time.time()
            detect_language = None if language == "auto" else language
            
            try:
                segments, info = whisper_model.transcribe(
                    audio_array,
                    language=detect_language,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "max_speech_duration_s": 30,
                        "min_silence_duration_ms": 100,
                        "speech_pad_ms": 30
                    }
                )
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"success": False, "error": f"Transcription failed: {str(e)}", "text": ""}
            
            results = []
            for segment in segments:
                try:
                    segment_dict = {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text.strip(),
                        "words": []
                    }
                    
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            try:
                                segment_dict["words"].append({
                                    "start": float(word.start),
                                    "end": float(word.end),
                                    "word": word.word,
                                    "probability": float(word.probability)
                                })
                            except:
                                continue
                    
                    results.append(segment_dict)
                except:
                    continue
            
            processing_time = time.time() - start_time
            audio_duration = getattr(info, 'duration', len(audio_array) / Config.SAMPLE_RATE)
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            return {
                "success": True,
                "language": info.language,
                "language_probability": float(info.language_probability),
                "duration": float(audio_duration),
                "processing_time": float(processing_time),
                "real_time_factor": float(rtf),
                "segments": results,
                "text": " ".join([seg["text"] for seg in results])
            }
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"success": False, "error": str(e), "text": ""}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            await websocket.send_text(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    global whisper_model
    
    logger.info("Starting Real-time Whisper Subtitles v2.2.2 (Memory Safe)...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()
    
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(3):
        try:
            whisper_model = WhisperManager.load_model()
            break
        except Exception as e:
            logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == 2:
                raise
            await asyncio.sleep(5)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Real-time Whisper Subtitles"
    })

@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    model_loaded = whisper_model is not None
    
    health_data = {
        "status": "healthy" if model_loaded else "loading",
        "version": "2.2.2",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
        "active_connections": len(manager.active_connections),
        "message": "Memory Safe - Double free error fixed",
        "fixes_applied": [
            "jemalloc memory allocator",
            "Safe library initialization",
            "Memory leak prevention",
            "Double free error fixed"
        ]
    }
    
    if gpu_available:
        try:
            health_data["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
        except:
            pass
    
    return UTF8JSONResponse(health_data)

@app.post("/api/transcribe")
async def transcribe_file(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form(Config.WHISPER_MODEL)
):
    temp_file_path = None
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        audio_array, sample_rate = sf.read(temp_file_path)
        
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        if sample_rate != Config.SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=Config.SAMPLE_RATE
            )
        
        audio_array = audio_array.astype(np.float32)
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        result = await WhisperManager.transcribe(audio_array, language)
        
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return UTF8JSONResponse(result)
        
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return UTF8JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if not AudioProcessor.detect_speech(data):
                continue
            
            audio_array = AudioProcessor.preprocess_audio(data)
            
            if len(audio_array) == 0 or len(audio_array) < Config.SAMPLE_RATE * 0.5:
                continue
            
            try:
                result = await WhisperManager.transcribe(audio_array)
                await manager.send_personal_message(result, websocket)
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": f"Transcription error: {str(e)}",
                    "text": ""
                }
                await manager.send_personal_message(error_result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/models")
async def get_available_models():
    models = [
        {"name": "tiny", "size": "39MB", "speed": "~32x", "description": "Fastest, lowest quality"},
        {"name": "base", "size": "74MB", "speed": "~16x", "description": "Balanced speed and quality"},
        {"name": "small", "size": "244MB", "speed": "~6x", "description": "Good quality, moderate speed"},
        {"name": "medium", "size": "769MB", "speed": "~2x", "description": "High quality, slower"},
        {"name": "large-v2", "size": "1550MB", "speed": "~1x", "description": "Best quality, slowest"},
        {"name": "large-v3", "size": "1550MB", "speed": "~1x", "description": "Latest, best quality"}
    ]
    return UTF8JSONResponse({"models": models})

@app.get("/api/languages")
async def get_supported_languages():
    languages = [
        {"code": "auto", "name": "Auto-detect"},
        {"code": "en", "name": "English"},
        {"code": "ja", "name": "Japanese"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ko", "name": "Korean"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"}
    ]
    return UTF8JSONResponse({"languages": languages})

@app.get("/api/stats")
async def get_system_stats():
    stats = {
        "gpu_available": torch.cuda.is_available(),
        "active_connections": len(manager.active_connections),
        "model_loaded": whisper_model is not None,
        "device": Config.DEVICE,
        "compute_type": Config.COMPUTE_TYPE
    }
    
    if torch.cuda.is_available():
        try:
            stats["gpu_stats"] = {
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                "device_name": torch.cuda.get_device_name(0)
            }
        except:
            pass
    
    return UTF8JSONResponse(stats)

if __name__ == "__main__":
    for path in [Config.STATIC_PATH, Config.TEMPLATE_PATH, Config.MODEL_PATH, Config.OUTPUT_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        "web_interface:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        access_log=True,
        log_level="info"
    )
