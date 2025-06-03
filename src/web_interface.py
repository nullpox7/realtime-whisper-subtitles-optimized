#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Main Web Interface (Fixed v2.0.3)
CUDA 12.9.0 + cuDNN optimized version - Encoding and audio processing issues fixed

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

# Log directory configuration - using /app/data/logs
log_dir = os.getenv('LOG_PATH', '/app/data/logs')
log_file_path = os.path.join(log_dir, 'whisper_app.log')

# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging with UTF-8 support and English messages
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
    description="Real-time speech recognition with faster-whisper",
    version="2.0.3"
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
    """Simplified audio processor without PyDub dependency"""
    
    @staticmethod
    def preprocess_audio(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> np.ndarray:
        """Process audio data without PyDub to avoid conversion errors"""
        try:
            if len(audio_data) == 0:
                logger.warning("Received empty audio data")
                return np.array([], dtype=np.float32)
            
            try:
                # Direct conversion from bytes to numpy array (16-bit PCM)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                # Normalize to [-1, 1] range
                audio_array = audio_array / 32768.0
                
                if len(audio_array) == 0:
                    logger.warning("Audio array is empty after conversion")
                    return np.array([], dtype=np.float32)
                
                # Handle NaN and infinite values
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Clip to valid range
                audio_array = np.clip(audio_array, -1.0, 1.0)
                
                logger.debug(f"Audio processed successfully: {len(audio_array)} samples")
                return audio_array
                
            except Exception as e:
                logger.error(f"Direct audio conversion failed: {e}")
                return np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def detect_speech(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> bool:
        """Simple energy-based speech detection (replacing WebRTC VAD)"""
        try:
            if len(audio_data) == 0:
                return False
            
            # Convert to audio array
            audio_array = AudioProcessor.preprocess_audio(audio_data, sample_rate)
            
            if len(audio_array) == 0:
                return False
            
            # Simple energy-based detection
            energy = np.mean(audio_array ** 2)
            threshold = 0.001  # Adjust as needed
            
            is_speech = energy > threshold
            logger.debug(f"Speech detection: energy={energy:.6f}, threshold={threshold:.6f}, speech={is_speech}")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return True  # Default to processing if detection fails

class WhisperManager:
    """Whisper model management"""
    
    @staticmethod
    def load_model(model_name: str = Config.WHISPER_MODEL) -> WhisperModel:
        """Load Whisper model"""
        try:
            model_path = Config.MODEL_PATH / "whisper" / model_name
            
            if model_path.exists():
                logger.info(f"Loading cached model from {model_path}")
                model = WhisperModel(
                    str(model_path),
                    device=Config.DEVICE,
                    compute_type=Config.COMPUTE_TYPE
                )
            else:
                logger.info(f"Downloading model: {model_name}")
                Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
                model = WhisperModel(
                    model_name,
                    device=Config.DEVICE,
                    compute_type=Config.COMPUTE_TYPE,
                    download_root=str(Config.MODEL_PATH / "whisper")
                )
            
            logger.info(f"Model loaded successfully: {model_name} on {Config.DEVICE}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    async def transcribe(audio_array: np.ndarray, language: str = "auto") -> Dict:
        """Transcribe audio using Whisper"""
        try:
            if whisper_model is None:
                raise ValueError("Whisper model not loaded")
            
            if len(audio_array) == 0:
                return {
                    "success": False,
                    "error": "Empty audio data",
                    "text": ""
                }
            
            start_time = time.time()
            
            # Set language to None for auto-detection if "auto" is specified
            detect_language = None if language == "auto" else language
            
            segments, info = whisper_model.transcribe(
                audio_array,
                language=detect_language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
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
                            except Exception as e:
                                logger.debug(f"Word processing error: {e}")
                                continue
                    
                    results.append(segment_dict)
                    
                except Exception as e:
                    logger.error(f"Segment processing error: {e}")
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
            logger.error(f"Transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            await websocket.send_text(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    global whisper_model
    
    logger.info("Starting Real-time Whisper Subtitles...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
    else:
        logger.warning("GPU not available, using CPU")
    
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        whisper_model = WhisperManager.load_model()
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "Real-time Whisper Subtitles"
        })
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        raise HTTPException(status_code=500, detail="Template rendering failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    model_loaded = whisper_model is not None
    
    return UTF8JSONResponse({
        "status": "healthy" if model_loaded else "loading",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
        "active_connections": len(manager.active_connections),
        "log_directory": log_dir,
        "log_file_exists": os.path.exists(log_file_path),
        "timestamp": time.time(),
        "message": "Real-time Whisper Subtitles is running"
    })

@app.post("/api/transcribe")
async def transcribe_file(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form(Config.WHISPER_MODEL)
):
    """File transcription endpoint"""
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Load audio using soundfile/librosa instead of PyDub
            audio_array, sample_rate = sf.read(temp_file_path)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample if necessary
            if sample_rate != Config.SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=Config.SAMPLE_RATE
                )
            
            # Ensure float32 and proper range
            audio_array = audio_array.astype(np.float32)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            result = await WhisperManager.transcribe(audio_array, language)
            os.unlink(temp_file_path)
            
            return UTF8JSONResponse(result)
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
        
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        return UTF8JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time transcription WebSocket endpoint"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Simple speech detection
            if not AudioProcessor.detect_speech(data):
                continue
            
            audio_array = AudioProcessor.preprocess_audio(data)
            
            if len(audio_array) == 0:
                continue
            
            # Skip very short audio
            if len(audio_array) < Config.SAMPLE_RATE * 0.5:
                continue
            
            result = await WhisperManager.transcribe(audio_array)
            await manager.send_personal_message(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/models")
async def get_available_models():
    """Get available Whisper models"""
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
    """Get supported languages"""
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

if __name__ == "__main__":
    # Create necessary directories
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
