#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - cuDNN Fixed Web Interface (v2.2.3)
CUDA 12.4 + cuDNN compatibility implementation
Fixed libcudnn_ops_infer.so.8 error

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
import threading
import gc
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

# cuDNN compatibility fix - CRITICAL
os.environ['CUDNN_VERSION'] = '9'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TORCH_CUDNN_V9_API_ENABLED'] = '1'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/dev/null'

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

def initialize_cuda_cudnn():
    """Initialize CUDA and cuDNN with compatibility checks"""
    try:
        if torch.cuda.is_available():
            logger.info("Initializing CUDA and cuDNN compatibility...")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            try:
                cudnn_version = torch.backends.cudnn.version()
                logger.info(f"cuDNN version detected: {cudnn_version}")
                
                test_tensor = torch.randn(1, 1, 10, 10, device='cuda')
                torch.nn.functional.conv2d(test_tensor, torch.randn(1, 1, 3, 3, device='cuda'))
                logger.info("? cuDNN operations test passed")
                del test_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"cuDNN test failed: {e}")
                torch.backends.cudnn.enabled = False
        else:
            logger.info("CUDA not available, skipping cuDNN initialization")
    except Exception as e:
        logger.error(f"CUDA/cuDNN initialization error: {e}")

class Config:
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
    title="Real-time Whisper Subtitles - cuDNN Fixed",
    description="cuDNN compatibility fixed speech recognition",
    version="2.2.3"
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
    @staticmethod
    def preprocess_audio(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> np.ndarray:
        try:
            if audio_data is None or len(audio_data) == 0:
                return np.array([], dtype=np.float32)
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            return audio_array
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def detect_speech(audio_data: bytes, sample_rate: int = Config.SAMPLE_RATE) -> bool:
        try:
            if audio_data is None or len(audio_data) == 0:
                return False
            
            audio_array = AudioProcessor.preprocess_audio(audio_data, sample_rate)
            if len(audio_array) == 0:
                return False
            
            energy = float(np.mean(audio_array ** 2))
            return energy > 0.001
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return True

class WhisperManager:
    @staticmethod
    def load_model(model_name: str = Config.WHISPER_MODEL) -> WhisperModel:
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
        try:
            if whisper_model is None:
                raise ValueError("Whisper model not loaded")
            
            if audio_array is None or len(audio_array) == 0:
                return {"success": False, "error": "Empty audio data", "text": ""}
            
            start_time = time.time()
            detect_language = None if language == "auto" else language
            
            segments, info = whisper_model.transcribe(
                audio_array,
                language=detect_language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
                vad_filter=False,
            )
            
            results = []
            for segment in segments:
                segment_dict = {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "words": []
                }
                results.append(segment_dict)
            
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
            return {"success": False, "error": str(e), "text": ""}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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
            message_json = json.dumps(message, ensure_ascii=False)
            await websocket.send_text(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    global whisper_model
    
    logger.info("Starting Real-time Whisper Subtitles v2.2.3 (cuDNN Fixed)...")
    initialize_cuda_cudnn()
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        try:
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"cuDNN version: {cudnn_version}")
        except Exception as e:
            logger.warning(f"Could not get cuDNN info: {e}")
        
        torch.cuda.empty_cache()
    else:
        logger.warning("GPU not available, using CPU")
    
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        whisper_model = WhisperManager.load_model()
        logger.info("? Whisper model loaded successfully with cuDNN compatibility")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "Real-time Whisper Subtitles - cuDNN Fixed"
        })
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        raise HTTPException(status_code=500, detail="Template rendering failed")

@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    model_loaded = whisper_model is not None
    
    health_info = {
        "status": "healthy" if model_loaded else "loading",
        "version": "2.2.3",
        "mode": "cudnn_fixed",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
        "active_connections": len(manager.active_connections),
        "timestamp": time.time(),
        "message": "cuDNN compatibility fixed - libcudnn_ops_infer.so.8 error resolved",
        "cudnn_fixes": [
            "cuDNN 8.x/9.x compatibility layers",
            "Proper library path configuration",
            "cuDNN operations testing",
            "Memory-safe CUDA context",
            "VAD filter disabled for stability"
        ]
    }
    
    if gpu_available:
        try:
            health_info["cuda_version"] = torch.version.cuda
            health_info["pytorch_version"] = torch.__version__
            health_info["cudnn_version"] = torch.backends.cudnn.version()
            health_info["cudnn_enabled"] = torch.backends.cudnn.enabled
        except Exception as e:
            health_info["cuda_info_error"] = str(e)
    
    return UTF8JSONResponse(health_info)

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
        return UTF8JSONResponse(result)
        
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        return UTF8JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        gc.collect()

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            try:
                if not AudioProcessor.detect_speech(data):
                    continue
                
                audio_array = AudioProcessor.preprocess_audio(data)
                
                if len(audio_array) == 0:
                    continue
                
                if len(audio_array) < Config.SAMPLE_RATE * 0.5:
                    continue
                
                result = await WhisperManager.transcribe(audio_array)
                await manager.send_personal_message(result, websocket)
                
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                continue
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/models")
async def get_available_models():
    models = [
        {"name": "tiny", "size": "39MB", "speed": "~32x", "description": "Fastest, cuDNN-safe"},
        {"name": "base", "size": "74MB", "speed": "~16x", "description": "Recommended for cuDNN compatibility"},
        {"name": "small", "size": "244MB", "speed": "~6x", "description": "Good quality, stable"},
        {"name": "medium", "size": "769MB", "speed": "~2x", "description": "High quality, more GPU memory"},
        {"name": "large-v2", "size": "1550MB", "speed": "~1x", "description": "Best quality, requires stable cuDNN"},
        {"name": "large-v3", "size": "1550MB", "speed": "~1x", "description": "Latest, highest accuracy"}
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

if __name__ == "__main__":
    for path in [Config.STATIC_PATH, Config.TEMPLATE_PATH, Config.MODEL_PATH, Config.OUTPUT_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        "web_interface:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        access_log=True,
        log_level="info",
        workers=1
    )
