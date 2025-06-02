/*
 * Real-time Whisper Subtitles - Frontend JavaScript Application (Fixed)
 * CUDA 12.9.0 + cuDNN optimized version - WebSocket audio streaming fixed
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
    
    console.log('WebSocket????:', wsUrl);
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function(event) {
        console.log('WebSocket????');
        updateStatus('connected', '????');
    };
    
    websocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleTranscriptionResult(data);
        } catch (error) {
            console.error('WebSocket??????????:', error);
        }
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket??');
        updateStatus('disconnected', '???????');
        
        // ?????????????
        if (!isRecording) {
            setTimeout(() => {
                console.log('WebSocket????...');
                initWebSocket();
            }, CONFIG.RECONNECT_INTERVAL);
        }
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket???:', error);
        updateStatus('disconnected', '???');
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
    console.log('????:', data);
    
    // ??????????????
    processingIndicator.classList.remove('show');
    
    if (data.success && data.text && data.text.trim()) {
        appendSubtitle(data.text.trim());
        updateStatistics(data);
    } else if (!data.success && data.error) {
        console.error('?????:', data.error);
        showNotification('?????: ' + data.error, 'error');
    }
}

/**
 * Append subtitle to display area
 */
function appendSubtitle(text) {
    const now = new Date().toLocaleTimeString('ja-JP');
    const subtitleEl = document.createElement('div');
    subtitleEl.className = 'subtitle-entry fade-in mb-2';
    subtitleEl.innerHTML = `
        <span class="text-muted small">[${now}]</span> 
        <span class="subtitle-text">${escapeHtml(text)}</span>
    `;
    
    // ?????????????????
    if (subtitleDisplay.querySelector('.text-center')) {
        subtitleDisplay.innerHTML = '';
    }
    
    subtitleDisplay.appendChild(subtitleEl);
    
    // ???????
    const subtitleEntries = subtitleDisplay.querySelectorAll('.subtitle-entry');
    if (subtitleEntries.length > CONFIG.MAX_SUBTITLE_HISTORY) {
        subtitleEntries[0].remove();
    }
    
    // ???????
    subtitleDisplay.scrollTop = subtitleDisplay.scrollHeight;
}

/**
 * Update statistics display
 */
function updateStatistics(data) {
    if (data.processing_time !== undefined) {
        processingTimeEl.textContent = `${data.processing_time.toFixed(2)}?`;
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
        console.log('????????...');
        
        // ????????????
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: CONFIG.SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        // AudioContext???
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.SAMPLE_RATE
        });
        
        const source = audioContext.createMediaStreamSource(audioStream);
        
        // ???????????????
        processor = audioContext.createScriptProcessor(CONFIG.AUDIO_BUFFER_SIZE, 1, 1);
        
        processor.onaudioprocess = function(event) {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                
                // float32?int16???
                const samples = new Int16Array(inputBuffer.length);
                for (let i = 0; i < inputBuffer.length; i++) {
                    const sample = Math.max(-1, Math.min(1, inputBuffer[i]));
                    samples[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                // ???????
                recordingBuffer.push(...samples);
                
                // ???????????????
                if (recordingBuffer.length >= CONFIG.SAMPLE_RATE * 2) { // 2??
                    const audioData = new Int16Array(recordingBuffer);
                    recordingBuffer = []; // ????????
                    
                    // ?????????????
                    processingIndicator.classList.add('show');
                    
                    // WebSocket???
                    websocket.send(audioData.buffer);
                }
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        // UI?????
        isRecording = true;
        recordBtn.innerHTML = '<i class="fas fa-stop"></i> ??';
        recordBtn.className = 'btn btn-success record-btn me-3';
        subtitleDisplay.classList.add('recording-active');
        recordingBuffer = []; // ????????
        
        showNotification('???????????', 'success');
        
    } catch (error) {
        console.error('???????:', error);
        showNotification('?????????????????????????????????', 'error');
    }
}

/**
 * Stop audio recording
 */
function stopRecording() {
    console.log('????????...');
    
    try {
        // ??????????
        if (recordingBuffer.length > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
            const audioData = new Int16Array(recordingBuffer);
            websocket.send(audioData.buffer);
            recordingBuffer = [];
        }
        
        // ?????????????
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        // ??????????????
        if (processor) {
            processor.disconnect();
            processor = null;
        }
        
        // ???????????????
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        // UI?????
        isRecording = false;
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> ??';
        recordBtn.className = 'btn btn-danger record-btn me-3';
        subtitleDisplay.classList.remove('recording-active');
        processingIndicator.classList.remove('show');
        
        showNotification('???????????', 'info');
        
    } catch (error) {
        console.error('???????:', error);
    }
}

/**
 * Clear subtitle display
 */
function clearSubtitles() {
    subtitleDisplay.innerHTML = `
        <div class="text-center text-muted">
            <i class="fas fa-microphone fa-3x mb-3"></i>
            <p>???????????????????????</p>
        </div>
    `;
    
    // ???????
    processingTimeEl.textContent = '-';
    detectedLangEl.textContent = '-';
    confidenceEl.textContent = '-';
    realtimeFactorEl.textContent = '-';
    
    console.log('??????????');
}

/**
 * Handle file upload
 */
async function handleFileUpload(file) {
    console.log('???????????:', file.name);
    
    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('language', languageSelect.value);
    formData.append('model', modelSelect.value);
    
    // ???????????
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
                <p><strong>????:</strong></p>
                <div class="bg-light p-3 rounded mb-3">${escapeHtml(result.text)}</div>
                <small class="text-muted">
                    ????: ${result.processing_time?.toFixed(2)}? | 
                    ??: ${result.language || 'N/A'} | 
                    ???: ${result.language_probability ? (result.language_probability * 100).toFixed(1) + '%' : 'N/A'}
                </small>
            `;
            fileResults.style.display = 'block';
            showNotification('?????????????', 'success');
        } else {
            throw new Error(result.error || '?????????');
        }
        
    } catch (error) {
        console.error('?????????????:', error);
        showNotification(`?????????: ${error.message}`, 'error');
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
    
    // 5???????
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
        console.log('???????:', data);
        
        if (!data.gpu_available) {
            showNotification('GPU ?????????CPU ????????????', 'warning');
        }
        
        if (!data.model_loaded) {
            showNotification('Whisper ?????????...', 'info');
        }
        
    } catch (error) {
        console.error('?????????:', error);
        showNotification('???????????????', 'error');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('Real-time Whisper Subtitles ?????...');
    
    // WebSocket??????
    initWebSocket();
    
    // ???????????????
    checkHealth();
    
    // ?????????
    recordBtn.addEventListener('click', function() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    // ??????????
    clearBtn.addEventListener('click', clearSubtitles);
    
    // ??????????????
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
    
    // ????????????
    languageSelect.addEventListener('change', () => {
        console.log('?????:', languageSelect.value);
    });
    
    modelSelect.addEventListener('change', () => {
        console.log('??????:', modelSelect.value);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    console.log('?????????????????...');
    
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
        console.log('????????????...');
    } else if (!document.hidden && websocket && websocket.readyState === WebSocket.CLOSED) {
        console.log('??????WebSocket???...');
        initWebSocket();
    }
});

// Handle window focus
window.addEventListener('focus', function() {
    if (websocket && websocket.readyState === WebSocket.CLOSED) {
        console.log('???????????WebSocket???...');
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

console.log('Real-time Whisper Subtitles JavaScript??????');
