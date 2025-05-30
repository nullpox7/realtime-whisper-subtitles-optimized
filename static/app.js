/*
 * Real-time Whisper Subtitles - Frontend JavaScript Application
 * CUDA 12.9.0 + cuDNN optimized version
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
    MAX_SUBTITLE_HISTORY: 50
};

/**
 * Initialize WebSocket connection
 */
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function(event) {
        console.log('WebSocket connected');
        updateStatus('connected', '????');
    };
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleTranscriptionResult(data);
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket disconnected');
        updateStatus('disconnected', '???????');
        
        // Auto-reconnect after specified interval
        setTimeout(() => {
            if (!isRecording) { // Only reconnect if not currently recording
                initWebSocket();
            }
        }, CONFIG.RECONNECT_INTERVAL);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateStatus('disconnected', '???');
    };
}

/**
 * Update status indicator
 * @param {string} status - Status type ('connected' or 'disconnected')
 * @param {string} text - Status text to display
 */
function updateStatus(status, text) {
    statusIndicator.className = `status-indicator status-${status}`;
    statusIndicator.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
}

/**
 * Handle transcription results from WebSocket
 * @param {Object} data - Transcription result data
 */
function handleTranscriptionResult(data) {
    console.log('Transcription result:', data);
    
    if (data.success && data.text && data.text.trim()) {
        appendSubtitle(data.text.trim());
        updateStatistics(data);
    } else if (!data.success && data.error) {
        console.error('Transcription error:', data.error);
        showNotification('???????: ' + data.error, 'error');
    }
}

/**
 * Append subtitle to display area
 * @param {string} text - Subtitle text to append
 */
function appendSubtitle(text) {
    const now = new Date().toLocaleTimeString('ja-JP');
    const subtitleEl = document.createElement('div');
    subtitleEl.className = 'subtitle-entry fade-in mb-2';
    subtitleEl.innerHTML = `
        <span class="text-muted small">[${now}]</span> 
        <span class="subtitle-text">${escapeHtml(text)}</span>
    `;
    
    // Clear placeholder content if exists
    if (subtitleDisplay.querySelector('.text-center')) {
        subtitleDisplay.innerHTML = '';
    }
    
    subtitleDisplay.appendChild(subtitleEl);
    
    // Limit subtitle history
    const subtitleEntries = subtitleDisplay.querySelectorAll('.subtitle-entry');
    if (subtitleEntries.length > CONFIG.MAX_SUBTITLE_HISTORY) {
        subtitleEntries[0].remove();
    }
    
    // Auto-scroll to bottom
    subtitleDisplay.scrollTop = subtitleDisplay.scrollHeight;
}

/**
 * Update statistics display
 * @param {Object} data - Statistics data
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
 * Start audio recording
 */
async function startRecording() {
    try {
        console.log('Starting audio recording...');
        
        // Request microphone access
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
        processor = audioContext.createScriptProcessor(CONFIG.CHUNK_SIZE, 1, 1);
        
        processor.onaudioprocess = function(event) {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                const samples = new Int16Array(inputBuffer.length);
                
                // Convert float32 to int16
                for (let i = 0; i < inputBuffer.length; i++) {
                    const sample = Math.max(-1, Math.min(1, inputBuffer[i]));
                    samples[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                websocket.send(samples.buffer);
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        // Update UI state
        isRecording = true;
        recordBtn.innerHTML = '<i class="fas fa-stop"></i> ????';
        recordBtn.className = 'btn btn-success record-btn me-3';
        subtitleDisplay.classList.add('recording-active');
        
        showNotification('?????????', 'success');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showNotification('???????????????????????????????????', 'error');
    }
}

/**
 * Stop audio recording
 */
function stopRecording() {
    console.log('Stopping audio recording...');
    
    try {
        // Stop all audio streams
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        // Disconnect audio processor
        if (processor) {
            processor.disconnect();
            processor = null;
        }
        
        // Close audio context
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        // Update UI state
        isRecording = false;
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> ????';
        recordBtn.className = 'btn btn-danger record-btn me-3';
        subtitleDisplay.classList.remove('recording-active');
        
        showNotification('?????????', 'info');
        
    } catch (error) {
        console.error('Error stopping recording:', error);
    }
}

/**
 * Clear subtitle display
 */
function clearSubtitles() {
    subtitleDisplay.innerHTML = `
        <div class="text-center text-muted">
            <i class="fas fa-microphone fa-3x mb-3"></i>
            <p>??????????????????????????</p>
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
 * @param {File} file - Audio file to upload
 */
async function handleFileUpload(file) {
    console.log('Uploading file:', file.name);
    
    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('language', languageSelect.value);
    formData.append('model', modelSelect.value);
    
    // Show loading state
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
                <p><strong>??????:</strong></p>
                <div class="bg-light p-3 rounded mb-3">${escapeHtml(result.text)}</div>
                <small class="text-muted">
                    ????: ${result.processing_time?.toFixed(2)}? | 
                    ????: ${result.language || 'N/A'} | 
                    ???: ${result.language_probability ? (result.language_probability * 100).toFixed(1) + '%' : 'N/A'}
                </small>
            `;
            fileResults.style.display = 'block';
            showNotification('?????????????', 'success');
        } else {
            throw new Error(result.error || '?????????');
        }
        
    } catch (error) {
        console.error('File upload error:', error);
        showNotification(`?????????: ${error.message}`, 'error');
    } finally {
        fileUploadArea.classList.remove('processing');
    }
}

/**
 * Show notification message
 * @param {string} message - Notification message
 * @param {string} type - Notification type ('success', 'error', 'info', 'warning')
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
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
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
            showNotification('GPU ?????????CPU ??????????', 'warning');
        }
        
        if (!data.model_loaded) {
            showNotification('Whisper ???????????...', 'info');
        }
        
    } catch (error) {
        console.error('Health check failed:', error);
        showNotification('????????????????', 'error');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Real-time Whisper Subtitles...');
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Check application health
    checkHealth();
    
    // Record button event
    recordBtn.addEventListener('click', function() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    // Clear button event
    clearBtn.addEventListener('click', clearSubtitles);
    
    // File upload events
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
                showNotification('???????????????', 'warning');
            }
        }
    });
    
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Language and model change events
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
        console.log('Tab hidden, maintaining recording...');
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

console.log('Real-time Whisper Subtitles JavaScript loaded successfully');
