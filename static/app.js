/*
 * Real-time Whisper Subtitles - Frontend JavaScript Application (Fixed)
 * CUDA 12.9.0 + cuDNN optimized version - Audio processing fixes
 * 
 * Author: Real-time Whisper Subtitles Team
 * License: MIT
 * Encoding: UTF-8
 */

// Global variables
let websocket = null;
let mediaRecorder = null;
let audioStream = null;
let isRecording = false;
let audioContext = null;
let processor = null;
let recordingBuffer = [];

// DOM elements
const recordBtn = document.getElementById('recordBtn');
const clearBtn = document.getElementById('clearBtn');
const subtitleDisplay = document.getElementById('subtitleDisplay');
const statusIndicator = document.getElementById('status');
const languageSelect = document.getElementById('languageSelect');
const modelSelect = document.getElementById('modelSelect');
const fileUploadArea = document.getElementById('fileUploadArea');
const audioFileInput = document.getElementById('audioFileInput');
const fileResults = document.getElementById('fileResults');
const fileTranscription = document.getElementById('fileTranscription');
const processingIndicator = document.getElementById('processingIndicator');

// Statistics elements
const processingTimeEl = document.getElementById('processingTime');
const detectedLangEl = document.getElementById('detectedLang');
const confidenceEl = document.getElementById('confidence');
const realtimeFactorEl = document.getElementById('realtimeFactor');

// Configuration
const CONFIG = {
    SAMPLE_RATE: 16000,
    CHUNK_SIZE: 4096,
    RECONNECT_INTERVAL: 3000,
    MAX_SUBTITLE_HISTORY: 50,
    AUDIO_BUFFER_SIZE: 8192
};

/**
 * Initialize WebSocket connection
 */
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime`;
    
    console.log('Connecting WebSocket:', wsUrl);
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function(event) {
        console.log('WebSocket connected');
        updateStatus('connected', 'Connected');
    };
    
    websocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleTranscriptionResult(data);
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket disconnected');
        updateStatus('disconnected', 'Disconnected');
        
        // Auto-reconnect if not recording
        if (!isRecording) {
            setTimeout(() => {
                console.log('Reconnecting WebSocket...');
                initWebSocket();
            }, CONFIG.RECONNECT_INTERVAL);
        }
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateStatus('disconnected', 'Error');
    };
}

/**
 * Update status indicator
 */
function updateStatus(status, text) {
    statusIndicator.className = `status-indicator status-${status}`;
    statusIndicator.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
}

/**
 * Handle transcription results from WebSocket
 */
function handleTranscriptionResult(data) {
    console.log('Transcription result:', data);
    
    // Hide processing indicator
    processingIndicator.classList.remove('show');
    
    if (data.success && data.text && data.text.trim()) {
        appendSubtitle(data.text.trim());
        updateStatistics(data);
    } else if (!data.success && data.error) {
        console.error('Transcription error:', data.error);
        showNotification('Transcription error: ' + data.error, 'error');
    }
}

/**
 * Append subtitle to display area
 */
function appendSubtitle(text) {
    const now = new Date().toLocaleTimeString();
    const subtitleEl = document.createElement('div');
    subtitleEl.className = 'subtitle-entry fade-in mb-2';
    subtitleEl.innerHTML = `
        <span class="text-muted small">[${now}]</span> 
        <span class="subtitle-text">${escapeHtml(text)}</span>
    `;
    
    // Clear placeholder if present
    if (subtitleDisplay.querySelector('.text-center')) {
        subtitleDisplay.innerHTML = '';
    }
    
    subtitleDisplay.appendChild(subtitleEl);
    
    // Limit history
    const subtitleEntries = subtitleDisplay.querySelectorAll('.subtitle-entry');
    if (subtitleEntries.length > CONFIG.MAX_SUBTITLE_HISTORY) {
        subtitleEntries[0].remove();
    }
    
    // Auto-scroll
    subtitleDisplay.scrollTop = subtitleDisplay.scrollHeight;
}

/**
 * Update statistics display
 */
function updateStatistics(data) {
    if (data.processing_time !== undefined) {
        processingTimeEl.textContent = `${data.processing_time.toFixed(2)}s`;
    }
    
    if (data.language) {
        detectedLangEl.textContent = data.language.toUpperCase();
    }
    
    if (data.language_probability !== undefined) {
        confidenceEl.textContent = `${(data.language_probability * 100).toFixed(1)}%`;
    }
    
    if (data.duration && data.processing_time) {
        const rtf = data.processing_time / data.duration;
        realtimeFactorEl.textContent = `${rtf.toFixed(2)}x`;
    }
}

/**
 * Start audio recording with improved WebSocket streaming
 */
async function startRecording() {
    try {
        console.log('Starting recording...');
        
        // Get audio stream
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: CONFIG.SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.SAMPLE_RATE
        });
        
        const source = audioContext.createMediaStreamSource(audioStream);
        
        // Create script processor for audio data
        processor = audioContext.createScriptProcessor(CONFIG.AUDIO_BUFFER_SIZE, 1, 1);
        
        processor.onaudioprocess = function(event) {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                
                // Convert float32 to int16
                const samples = new Int16Array(inputBuffer.length);
                for (let i = 0; i < inputBuffer.length; i++) {
                    const sample = Math.max(-1, Math.min(1, inputBuffer[i]));
                    samples[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                // Add to buffer
                recordingBuffer.push(...samples);
                
                // Send when buffer is large enough
                if (recordingBuffer.length >= CONFIG.SAMPLE_RATE * 2) { // 2 seconds
                    const audioData = new Int16Array(recordingBuffer);
                    recordingBuffer = []; // Clear buffer
                    
                    // Show processing indicator
                    processingIndicator.classList.add('show');
                    
                    // Send to WebSocket
                    websocket.send(audioData.buffer);
                }
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        // Update UI
        isRecording = true;
        recordBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
        recordBtn.className = 'btn btn-success record-btn me-3';
        subtitleDisplay.classList.add('recording-active');
        recordingBuffer = []; // Clear buffer
        
        showNotification('Recording started successfully', 'success');
        
    } catch (error) {
        console.error('Recording start error:', error);
        showNotification('Failed to start recording. Please check microphone permissions.', 'error');
    }
}

/**
 * Stop audio recording
 */
function stopRecording() {
    console.log('Stopping recording...');
    
    try {
        // Send remaining buffer
        if (recordingBuffer.length > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
            const audioData = new Int16Array(recordingBuffer);
            websocket.send(audioData.buffer);
            recordingBuffer = [];
        }
        
        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        // Disconnect processor
        if (processor) {
            processor.disconnect();
            processor = null;
        }
        
        // Close audio context
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        // Update UI
        isRecording = false;
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordBtn.className = 'btn btn-danger record-btn me-3';
        subtitleDisplay.classList.remove('recording-active');
        processingIndicator.classList.remove('show');
        
        showNotification('Recording stopped', 'info');
        
    } catch (error) {
        console.error('Recording stop error:', error);
    }
}

/**
 * Clear subtitle display
 */
function clearSubtitles() {
    subtitleDisplay.innerHTML = `
        <div class="text-center text-muted">
            <i class="fas fa-microphone fa-3x mb-3"></i>
            <p>Click "Start Recording" to begin real-time transcription</p>
        </div>
    `;
    
    // Reset statistics
    processingTimeEl.textContent = '-';
    detectedLangEl.textContent = '-';
    confidenceEl.textContent = '-';
    realtimeFactorEl.textContent = '-';
    
    console.log('Subtitles cleared');
}

/**
 * Handle file upload
 */
async function handleFileUpload(file) {
    console.log('Uploading file:', file.name);
    
    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('language', languageSelect.value);
    formData.append('model', modelSelect.value);
    
    // Show processing state
    fileUploadArea.classList.add('processing');
    fileResults.style.display = 'none';
    
    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            fileTranscription.innerHTML = `
                <p><strong>Transcription:</strong></p>
                <div class="bg-light p-3 rounded mb-3">${escapeHtml(result.text)}</div>
                <small class="text-muted">
                    Processing time: ${result.processing_time?.toFixed(2)}s | 
                    Language: ${result.language || 'N/A'} | 
                    Confidence: ${result.language_probability ? (result.language_probability * 100).toFixed(1) + '%' : 'N/A'}
                </small>
            `;
            fileResults.style.display = 'block';
            showNotification('File transcription completed', 'success');
        } else {
            throw new Error(result.error || 'Transcription failed');
        }
        
    } catch (error) {
        console.error('File upload error:', error);
        showNotification(`Upload failed: ${error.message}`, 'error');
    } finally {
        fileUploadArea.classList.remove('processing');
    }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    const alertClass = {
        success: 'alert-success',
        error: 'alert-danger',
        info: 'alert-info',
        warning: 'alert-warning'
    }[type] || 'alert-info';
    
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Check application health
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Health check:', data);
        
        if (!data.gpu_available) {
            showNotification('GPU not available, using CPU processing', 'warning');
        }
        
        if (!data.model_loaded) {
            showNotification('Whisper model loading...', 'info');
        }
        
    } catch (error) {
        console.error('Health check failed:', error);
        showNotification('Server connection failed', 'error');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('Real-time Whisper Subtitles loaded...');
    
    // Initialize WebSocket
    initWebSocket();
    
    // Check health
    checkHealth();
    
    // Recording controls
    recordBtn.addEventListener('click', function() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    // Clear button
    clearBtn.addEventListener('click', clearSubtitles);
    
    // File upload handlers
    fileUploadArea.addEventListener('click', () => audioFileInput.click());
    
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadArea.classList.add('dragover');
    });
    
    fileUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
    });
    
    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('audio/')) {
                handleFileUpload(file);
            } else {
                showNotification('Please select an audio file', 'warning');
            }
        }
    });
    
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Settings change handlers
    languageSelect.addEventListener('change', () => {
        console.log('Language changed to:', languageSelect.value);
    });
    
    modelSelect.addEventListener('change', () => {
        console.log('Model changed to:', modelSelect.value);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    console.log('Page unloading, cleaning up...');
    
    if (isRecording) {
        stopRecording();
    }
    
    if (websocket) {
        websocket.close();
    }
});

// Handle visibility change (tab switching)
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isRecording) {
        console.log('Tab hidden while recording...');
    } else if (!document.hidden && websocket && websocket.readyState === WebSocket.CLOSED) {
        console.log('Tab visible, reconnecting WebSocket...');
        initWebSocket();
    }
});

// Handle window focus
window.addEventListener('focus', function() {
    if (websocket && websocket.readyState === WebSocket.CLOSED) {
        console.log('Window focused, reconnecting WebSocket...');
        initWebSocket();
    }
});

// Export for debugging (development only)
if (typeof window !== 'undefined') {
    window.whisperApp = {
        websocket,
        isRecording,
        startRecording,
        stopRecording,
        clearSubtitles,
        checkHealth
    };
}

console.log('Real-time Whisper Subtitles JavaScript loaded');
