/*
 * Real-time Whisper Subtitles - Stream Edition JavaScript
 * Optimized for live streaming with microphone device selection
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
let availableMicrophones = [];
let selectedMicrophoneId = null;
let currentSubtitleText = '';
let isFullscreen = false;

// DOM elements
const recordBtn = document.getElementById('recordBtn');
const clearBtn = document.getElementById('clearBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');
const subtitleDisplay = document.getElementById('subtitleDisplay');
const fullscreenSubtitle = document.getElementById('fullscreenSubtitle');
const fullscreenText = document.getElementById('fullscreenText');
const statusIndicator = document.getElementById('status');
const microphoneSelect = document.getElementById('microphoneSelect');
const languageSelect = document.getElementById('languageSelect');
const modelSelect = document.getElementById('modelSelect');
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
    AUDIO_BUFFER_SIZE: 8192,
    SUBTITLE_FADE_DURATION: 300
};

/**
 * Initialize application
 */
async function initApp() {
    console.log('Initializing Stream Subtitles...');
    
    // Initialize WebSocket
    initWebSocket();
    
    // Load available microphones
    await loadMicrophones();
    
    // Check application health
    checkHealth();
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
}

/**
 * Load available microphone devices
 */
async function loadMicrophones() {
    try {
        // Request permission first
        await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Get all media devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableMicrophones = devices.filter(device => device.kind === 'audioinput');
        
        // Clear existing options
        microphoneSelect.innerHTML = '';
        
        if (availableMicrophones.length === 0) {
            microphoneSelect.innerHTML = '<option value="">No microphones found</option>';
            return;
        }
        
        // Add default option
        microphoneSelect.innerHTML = '<option value="">Default Microphone</option>';
        
        // Add each microphone
        availableMicrophones.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Microphone ${index + 1}`;
            microphoneSelect.appendChild(option);
        });
        
        console.log(`Found ${availableMicrophones.length} microphone(s)`);
        
    } catch (error) {
        console.error('Failed to load microphones:', error);
        microphoneSelect.innerHTML = '<option value="">Permission denied</option>';
        showNotification('Microphone access denied. Please grant permission.', 'error');
    }
}

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
        updateSubtitle(data.text.trim());
        updateStatistics(data);
    } else if (!data.success && data.error) {
        console.error('Transcription error:', data.error);
        showNotification('Transcription error: ' + data.error, 'error');
    }
}

/**
 * Update subtitle display (replaces previous text)
 */
function updateSubtitle(text) {
    currentSubtitleText = text;
    
    // Update main display
    const subtitleEl = document.createElement('div');
    subtitleEl.className = 'subtitle-text fade-in';
    subtitleEl.textContent = text;
    
    // Clear and add new subtitle
    subtitleDisplay.innerHTML = '';
    subtitleDisplay.appendChild(subtitleEl);
    
    // Update fullscreen display if active
    if (isFullscreen) {
        const fullscreenEl = document.createElement('div');
        fullscreenEl.className = 'subtitle-text fade-in';
        fullscreenEl.textContent = text;
        
        fullscreenText.innerHTML = '';
        fullscreenText.appendChild(fullscreenEl);
    }
    
    console.log('Subtitle updated:', text);
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
        
        // Show performance indicator
        const rtfElement = realtimeFactorEl.parentElement;
        if (rtf > 1.0) {
            rtfElement.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)';
        } else if (rtf > 0.5) {
            rtfElement.style.background = 'linear-gradient(135deg, #ffa726 0%, #ff9800 100%)';
        } else {
            rtfElement.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        }
    }
}

/**
 * Start audio recording with microphone selection
 */
async function startRecording() {
    try {
        console.log('Starting recording...');
        
        // Get selected microphone ID
        selectedMicrophoneId = microphoneSelect.value || null;
        
        // Audio constraints
        const audioConstraints = {
            sampleRate: CONFIG.SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
        };
        
        // Add device ID if specific microphone selected
        if (selectedMicrophoneId) {
            audioConstraints.deviceId = { exact: selectedMicrophoneId };
        }
        
        // Get audio stream
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: audioConstraints
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
        
        // Show recording indicator in subtitle area
        updateSubtitle('? Recording... Speak now!');
        
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
        
        // Show waiting message
        clearSubtitles();
        
        showNotification('Recording stopped', 'info');
        
    } catch (error) {
        console.error('Recording stop error:', error);
    }
}

/**
 * Clear subtitle display
 */
function clearSubtitles() {
    currentSubtitleText = '';
    
    // Clear main display
    subtitleDisplay.innerHTML = `
        <div class="waiting-text">
            <i class="fas fa-microphone-slash"></i><br>
            Click "Start Recording" to begin live subtitles
        </div>
    `;
    
    // Clear fullscreen display
    if (isFullscreen) {
        fullscreenText.innerHTML = `
            <div class="waiting-text">Ready for live subtitles...</div>
        `;
    }
    
    // Reset statistics
    processingTimeEl.textContent = '-';
    detectedLangEl.textContent = '-';
    confidenceEl.textContent = '-';
    realtimeFactorEl.textContent = '-';
    
    console.log('Subtitles cleared');
}

/**
 * Toggle fullscreen mode
 */
function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    
    if (isFullscreen) {
        // Show fullscreen subtitle
        fullscreenSubtitle.style.display = 'flex';
        fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i> Exit Fullscreen';
        
        // Update fullscreen text with current subtitle
        if (currentSubtitleText) {
            fullscreenText.innerHTML = `<div class="subtitle-text">${escapeHtml(currentSubtitleText)}</div>`;
        } else {
            fullscreenText.innerHTML = '<div class="waiting-text">Ready for live subtitles...</div>';
        }
        
    } else {
        // Hide fullscreen subtitle
        fullscreenSubtitle.style.display = 'none';
        fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i> Fullscreen';
    }
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // F key for fullscreen toggle
        if (event.key === 'f' || event.key === 'F') {
            event.preventDefault();
            toggleFullscreen();
        }
        
        // Space key for record toggle (when not in input fields)
        if (event.code === 'Space' && !event.target.matches('input, select, textarea')) {
            event.preventDefault();
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        
        // Escape key to exit fullscreen
        if (event.key === 'Escape' && isFullscreen) {
            toggleFullscreen();
        }
        
        // C key to clear subtitles
        if (event.key === 'c' || event.key === 'C') {
            if (!event.target.matches('input, select, textarea')) {
                event.preventDefault();
                clearSubtitles();
            }
        }
    });
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
    console.log('Stream Subtitles loading...');
    
    // Initialize application
    initApp();
    
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
    
    // Fullscreen button
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    
    // Settings change handlers
    microphoneSelect.addEventListener('change', () => {
        console.log('Microphone changed to:', microphoneSelect.value);
        if (isRecording) {
            // Restart recording with new microphone
            stopRecording();
            setTimeout(() => startRecording(), 500);
        }
    });
    
    languageSelect.addEventListener('change', () => {
        console.log('Language changed to:', languageSelect.value);
    });
    
    modelSelect.addEventListener('change', () => {
        console.log('Model changed to:', modelSelect.value);
    });
    
    // Fullscreen subtitle click to exit
    fullscreenSubtitle.addEventListener('click', function() {
        if (isFullscreen) {
            toggleFullscreen();
        }
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
    window.streamSubtitles = {
        websocket,
        isRecording,
        isFullscreen,
        startRecording,
        stopRecording,
        clearSubtitles,
        toggleFullscreen,
        availableMicrophones,
        checkHealth
    };
}

console.log('Stream Subtitles JavaScript loaded - Press F for fullscreen, Space to record');