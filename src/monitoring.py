#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Whisper Subtitles - Monitoring and Metrics Module
Prometheus metrics collection and system monitoring

Author: Real-time Whisper Subtitles Team
License: MIT
Encoding: UTF-8
"""

import logging
import time
import threading
from typing import Dict, Optional, List
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import json

logger = logging.getLogger(__name__)

class WhisperMetrics:
    """Prometheus metrics collector for Whisper Subtitles system"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._lock = threading.RLock()
        
        # Application metrics
        self.transcription_requests = Counter(
            'whisper_transcription_requests_total',
            'Total number of transcription requests',
            ['model', 'language', 'status'],
            registry=self.registry
        )
        
        self.transcription_duration = Histogram(
            'whisper_transcription_duration_seconds',
            'Time spent on transcription',
            ['model', 'language'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.audio_duration = Histogram(
            'whisper_audio_duration_seconds',
            'Duration of processed audio',
            ['model'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
            registry=self.registry
        )
        
        self.real_time_factor = Histogram(
            'whisper_real_time_factor',
            'Real-time factor (processing_time / audio_duration)',
            ['model'],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
            registry=self.registry
        )
        
        self.websocket_connections = Gauge(
            'whisper_websocket_connections_active',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            'whisper_model_load_duration_seconds',
            'Time to load Whisper models',
            ['model', 'device'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'whisper_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'whisper_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'whisper_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'whisper_gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'type'],
            registry=self.registry
        )
        
        # Audio processing metrics
        self.audio_chunks_processed = Counter(
            'whisper_audio_chunks_processed_total',
            'Total audio chunks processed',
            ['status'],
            registry=self.registry
        )
        
        self.vad_detections = Counter(
            'whisper_vad_detections_total',
            'Total VAD speech detections',
            ['result'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'whisper_errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'whisper_application_info',
            'Application information',
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'whisper_model_info',
            'Model information',
            registry=self.registry
        )
        
        logger.info("WhisperMetrics initialized")
    
    def record_transcription_request(
        self, 
        model: str, 
        language: str, 
        status: str,
        duration: float = 0.0,
        audio_duration: float = 0.0
    ):
        """Record a transcription request"""
        with self._lock:
            self.transcription_requests.labels(
                model=model,
                language=language,
                status=status
            ).inc()
            
            if status == 'success' and duration > 0:
                self.transcription_duration.labels(
                    model=model,
                    language=language
                ).observe(duration)
                
                if audio_duration > 0:
                    self.audio_duration.labels(model=model).observe(audio_duration)
                    rtf = duration / audio_duration
                    self.real_time_factor.labels(model=model).observe(rtf)
    
    def record_model_load(self, model: str, device: str, duration: float):
        """Record model loading time"""
        with self._lock:
            self.model_load_duration.labels(
                model=model,
                device=device
            ).observe(duration)
    
    def update_websocket_connections(self, count: int):
        """Update active WebSocket connections count"""
        self.websocket_connections.set(count)
    
    def record_audio_chunk(self, status: str = 'processed'):
        """Record audio chunk processing"""
        self.audio_chunks_processed.labels(status=status).inc()
    
    def record_vad_detection(self, result: str):
        """Record VAD detection result"""
        self.vad_detections.labels(result=result).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record an error"""
        self.errors.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.labels(type='total').set(memory.total)
            self.memory_usage.labels(type='available').set(memory.available)
            self.memory_usage.labels(type='used').set(memory.used)
            
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # GPU utilization
                    gpu_util = torch.cuda.utilization(i)
                    self.gpu_usage.labels(gpu_id=str(i)).set(gpu_util)
                    
                    # GPU memory
                    gpu_memory = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    total = gpu_memory.total_memory
                    
                    self.gpu_memory.labels(gpu_id=str(i), type='total').set(total)
                    self.gpu_memory.labels(gpu_id=str(i), type='allocated').set(allocated)
                    self.gpu_memory.labels(gpu_id=str(i), type='reserved').set(reserved)
                    
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def set_app_info(self, version: str, model: str, device: str):
        """Set application information"""
        self.app_info.info({
            'version': version,
            'default_model': model,
            'device': device,
            'cuda_available': str(torch.cuda.is_available()),
            'cuda_version': torch.version.cuda or 'N/A',
            'pytorch_version': torch.__version__
        })
    
    def set_model_info(self, model_name: str, info_dict: Dict):
        """Set model information"""
        self.model_info.info({
            'model_name': model_name,
            **{k: str(v) for k, v in info_dict.items()}
        })
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class SystemMonitor:
    """System resource monitor with alerts"""
    
    def __init__(self, metrics: WhisperMetrics, update_interval: float = 30.0):
        self.metrics = metrics
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.gpu_memory_threshold = 90.0
        
        logger.info("SystemMonitor initialized")
    
    def start(self):
        """Start the monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("SystemMonitor started")
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info("SystemMonitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.metrics.update_system_metrics()
                self._check_alerts()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5.0)
    
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            # CPU alert
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.cpu_threshold:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory alert
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            # GPU memory alert
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    usage_percent = (allocated_memory / total_memory) * 100
                    
                    if usage_percent > self.gpu_memory_threshold:
                        logger.warning(f"High GPU memory usage on GPU {i}: {usage_percent:.1f}%")
                        
        except Exception as e:
            logger.error(f"Alert check error: {e}")

class PerformanceProfiler:
    """Performance profiling and analysis"""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        return ProfileContext(self, name)
    
    def record_time(self, name: str, duration: float):
        """Record execution time for a named operation"""
        with self._lock:
            if name not in self.profiles:
                self.profiles[name] = []
            self.profiles[name].append(duration)
            
            # Keep only last 1000 measurements
            if len(self.profiles[name]) > 1000:
                self.profiles[name] = self.profiles[name][-1000:]
    
    def get_stats(self, name: str) -> Optional[Dict]:
        """Get statistics for a named operation"""
        with self._lock:
            if name not in self.profiles or not self.profiles[name]:
                return None
            
            times = self.profiles[name]
            return {
                'count': len(times),
                'min': min(times),
                'max': max(times),
                'mean': sum(times) / len(times),
                'median': sorted(times)[len(times) // 2],
                'p95': sorted(times)[int(len(times) * 0.95)],
                'p99': sorted(times)[int(len(times) * 0.99)]
            }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all operations"""
        return {name: self.get_stats(name) for name in self.profiles.keys()}

class ProfileContext:
    """Context manager for profiling"""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.profiler.record_time(self.name, duration)

# Global instances
_global_metrics: Optional[WhisperMetrics] = None
_global_monitor: Optional[SystemMonitor] = None
_global_profiler: Optional[PerformanceProfiler] = None

def get_metrics() -> WhisperMetrics:
    """Get or create global metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = WhisperMetrics()
    return _global_metrics

def get_monitor() -> SystemMonitor:
    """Get or create global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor(get_metrics())
    return _global_monitor

def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def start_monitoring():
    """Start the global monitoring system"""
    monitor = get_monitor()
    monitor.start()
    logger.info("Global monitoring started")

def stop_monitoring():
    """Stop the global monitoring system"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop()
    logger.info("Global monitoring stopped")

def cleanup_monitoring():
    """Cleanup monitoring resources"""
    global _global_metrics, _global_monitor, _global_profiler
    
    stop_monitoring()
    
    _global_metrics = None
    _global_monitor = None
    _global_profiler = None
    
    logger.info("Monitoring resources cleaned up")

class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection"""
    
    def __init__(self, metrics: WhisperMetrics):
        self.metrics = metrics
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record request metrics
            method = request.method
            path = request.url.path
            status_code = response.status_code
            
            # You can customize this based on your needs
            if path.startswith('/api/transcribe'):
                # This would be recorded separately in the transcription handler
                pass
            elif path.startswith('/ws/'):
                # WebSocket metrics are handled separately
                pass
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_error(
                error_type=type(e).__name__,
                component='middleware'
            )
            raise

# Decorator for automatic function profiling
def profile_function(name: Optional[str] = None):
    """Decorator to automatically profile function execution"""
    def decorator(func):
        profile_name = name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile(profile_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Decorator for automatic metrics recording
def record_metrics(model: str = 'unknown', language: str = 'unknown'):
    """Decorator to automatically record transcription metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                metrics.record_transcription_request(
                    model=model,
                    language=language,
                    status='success',
                    duration=duration
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure
                metrics.record_transcription_request(
                    model=model,
                    language=language,
                    status='error',
                    duration=duration
                )
                
                metrics.record_error(
                    error_type=type(e).__name__,
                    component='transcription'
                )
                
                raise
        
        return wrapper
    return decorator
