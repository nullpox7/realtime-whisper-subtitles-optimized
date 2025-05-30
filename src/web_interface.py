#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Main Web Interface
CUDA 12.9.0 + cuDNN optimized version

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
import webrtcvad
from pydub import AudioSegment
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/whisper_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Whisper settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    # Audio settings
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    VAD_MODE = int(os.getenv("VAD_MODE", 3))  # 0-3, aggressive level
    
    # Performance settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    
    # Paths
    MODEL_PATH = Path("/app/models")
    OUTPUT_PATH = Path("/app/outputs")
    STATIC_PATH = Path("/app/static")
    TEMPLATE_PATH = Path("/app/templates")

# Initialize FastAPI application
app = FastAPI(
    title="Real-time Whisper Subtitles",
    description="Real-time speech recognition with faster-whisper",
    version="2.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(Config.STATIC_PATH)), name="static")
templates = Jinja2Templates(directory=str(Config.TEMPLATE_PATH))

# Global variables
whisper_model: Optional[WhisperModel] = None
vad = webrtcvad.Vad(Config.VAD_MODE)
active_connections: List[WebSocket] = []

class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def preprocess_audio(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> np.ndarray:
        """Preprocess audio data for Whisper"""
        try:
            # Convert bytes to numpy array
            audio_segment = AudioSegment.from_raw(
                audio_data,
                sample_width=2,  # 16-bit
                frame_rate=sample_rate,
                channels=1
            )
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            return audio_array
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([])
    
    @staticmethod
    def detect_speech(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> bool:
        """Detect speech using WebRTC VAD"""
        try:
            if sample_rate not in [8000, 16000, 32000, 48000]:
                # Resample to supported rate
                sample_rate = 16000
            
            frame_duration = 30  # ms
            frame_size = int(sample_rate * frame_duration / 1000)
            
            if len(audio_data) < frame_size * 2:  # 16-bit samples
                return False
            
            return vad.is_speech(audio_data[:frame_size * 2], sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to processing if VAD fails

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
    async def transcribe(audio_array: np.ndarray, language: str = "ja") -> Dict:
        """Transcribe audio using Whisper"""
        try:
            if whisper_model is None:
                raise ValueError("Whisper model not loaded")
            
            start_time = time.time()
            
            # Transcribe
            segments, info = whisper_model.transcribe(
                audio_array,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=True
            )
            
            # Process segments
            results = []
            for segment in segments:
                results.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ] if hasattr(segment, 'words') and segment.words else []
                })
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "processing_time": processing_time,
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

# WebSocket connection manager
class ConnectionManager:
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
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    global whisper_model
    
    logger.info("Starting Real-time Whisper Subtitles...")
    
    # Check GPU availability
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
    
    # Create directories
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load Whisper model
    try:
        whisper_model = WhisperManager.load_model()
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Real-time Whisper Subtitles"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    model_loaded = whisper_model is not None
    
    return JSONResponse({
        "status": "healthy" if model_loaded else "loading",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
        "active_connections": len(manager.active_connections),
        "timestamp": time.time()
    })

@app.post("/api/transcribe")
async def transcribe_file(
    audio_file: UploadFile = File(...),
    language: str = Form("ja"),
    model: str = Form(Config.WHISPER_MODEL)
):
    """Transcribe uploaded audio file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load and preprocess audio
        audio_array, sample_rate = sf.read(temp_file_path)
        
        # Ensure mono and correct sample rate
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        if sample_rate != Config.SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=Config.SAMPLE_RATE
            )
        
        # Transcribe
        result = await WhisperManager.transcribe(audio_array, language)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time transcription WebSocket"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Check if speech is detected
            if not AudioProcessor.detect_speech(data):
                continue
            
            # Preprocess audio
            audio_array = AudioProcessor.preprocess_audio(data)
            
            if len(audio_array) == 0:
                continue
            
            # Transcribe
            result = await WhisperManager.transcribe(audio_array)
            
            # Send result
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
        {"name": "tiny", "size": "39MB", "speed": "~32x"},
        {"name": "base", "size": "74MB", "speed": "~16x"},
        {"name": "small", "size": "244MB", "speed": "~6x"},
        {"name": "medium", "size": "769MB", "speed": "~2x"},
        {"name": "large-v2", "size": "1550MB", "speed": "~1x"},
        {"name": "large-v3", "size": "1550MB", "speed": "~1x"}
    ]
    
    return JSONResponse({"models": models})

@app.get("/api/languages")
async def get_supported_languages():
    """Get supported languages"""
    languages = [
        {"code": "ja", "name": "Japanese"},
        {"code": "en", "name": "English"},
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
    
    return JSONResponse({"languages": languages})

if __name__ == "__main__":
    # Create necessary directories
    for path in [Config.STATIC_PATH, Config.TEMPLATE_PATH, Config.MODEL_PATH, Config.OUTPUT_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "web_interface:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        access_log=True,
        log_level="info"
    )
