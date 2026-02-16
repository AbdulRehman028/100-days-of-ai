/**
 * Day 54 - Real-time Emotion Detection
 * Frontend JavaScript for webcam capture and emotion display
 */

// DOM Elements
const webcamBtn = document.getElementById('webcamBtn');
const uploadBtn = document.getElementById('uploadBtn');
const webcamSection = document.getElementById('webcamSection');
const uploadSection = document.getElementById('uploadSection');

const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const autoDetectCheckbox = document.getElementById('autoDetect');

const fileInput = document.getElementById('fileInput');
const uploadedImage = document.getElementById('uploadedImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadOverlay = document.getElementById('uploadOverlay');
const analyzeBtn = document.getElementById('analyzeBtn');

const statusBar = document.getElementById('statusBar');
const mainEmotion = document.getElementById('mainEmotion');
const emotionEmoji = document.getElementById('emotionEmoji');
const emotionLabel = document.getElementById('emotionLabel');
const emotionConfidence = document.getElementById('emotionConfidence');
const emotionBars = document.getElementById('emotionBars');
const faceCount = document.getElementById('faceCount');
const faceCountNum = document.getElementById('faceCountNum');

const processCanvas = document.getElementById('processCanvas');

// State
let stream = null;
let autoDetectInterval = null;
let isProcessing = false;

// Emotion colors for bars
const emotionColors = {
    'Angry': 'emotion-angry',
    'Disgust': 'emotion-disgust',
    'Fear': 'emotion-fear',
    'Happy': 'emotion-happy',
    'Sad': 'emotion-sad',
    'Surprise': 'emotion-surprise',
    'Neutral': 'emotion-neutral'
};

// Results panel element
const resultsPanel = document.getElementById('resultsPanel');

// Track current mode
let isWebcamMode = true;

// Tab switching
webcamBtn.addEventListener('click', () => {
    isWebcamMode = true;
    webcamBtn.classList.add('active-tab', 'bg-purple-600');
    webcamBtn.classList.remove('bg-gray-700');
    uploadBtn.classList.remove('active-tab', 'bg-purple-600');
    uploadBtn.classList.add('bg-gray-700');
    webcamSection.classList.remove('hidden');
    uploadSection.classList.add('hidden');
    // Hide results panel in webcam mode
    if (resultsPanel) resultsPanel.classList.add('hidden');
});

uploadBtn.addEventListener('click', () => {
    isWebcamMode = false;
    uploadBtn.classList.add('active-tab', 'bg-purple-600');
    uploadBtn.classList.remove('bg-gray-700');
    webcamBtn.classList.remove('active-tab', 'bg-purple-600');
    webcamBtn.classList.add('bg-gray-700');
    uploadSection.classList.remove('hidden');
    webcamSection.classList.add('hidden');
    stopAutoDetect();
    // Show results panel in upload mode
    if (resultsPanel) resultsPanel.classList.remove('hidden');
});

// Webcam functions
async function startCamera() {
    try {
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            updateStatus('Camera not supported. Please open in Chrome/Edge/Firefox (not VS Code browser).', 'error');
            return;
        }

        // Check if we're in a secure context
        if (!window.isSecureContext) {
            updateStatus('Camera requires HTTPS or localhost. Current URL is not secure.', 'error');
            return;
        }

        // Check current permission status first
        if (navigator.permissions && navigator.permissions.query) {
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'camera' });
                console.log('Camera permission status:', permissionStatus.state);
                
                if (permissionStatus.state === 'denied') {
                    updateStatus('Camera blocked by browser. Click the 🔒 lock icon in address bar → Site settings → Camera → Allow, then refresh.', 'error');
                    return;
                }
            } catch (e) {
                // Permission query not supported, continue anyway
                console.log('Permission query not supported:', e);
            }
        }

        updateStatus('Requesting camera access...', 'loading');
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        
        webcam.srcObject = stream;
        webcamPlaceholder.classList.add('hidden');
        startCameraBtn.classList.add('hidden');
        stopCameraBtn.classList.remove('hidden');
        captureBtn.classList.remove('hidden');
        
        // Auto-enable continuous detection
        autoDetectCheckbox.checked = true;
        startAutoDetect();
        
        updateStatus('Camera started with real-time detection enabled!', 'success');
    } catch (err) {
        console.error('Camera error:', err);
        let errorMsg = 'Failed to access camera. ';
        if (err.name === 'NotAllowedError') {
            errorMsg += 'Permission denied. Click 🔒 lock icon → Site settings → Camera → Allow, then refresh page.';
        } else if (err.name === 'NotFoundError') {
            errorMsg += 'No camera found. Please connect a webcam.';
        } else if (err.name === 'NotReadableError') {
            errorMsg += 'Camera in use by another app. Close other apps and retry.';
        } else if (err.name === 'AbortError') {
            errorMsg += 'Camera request was aborted. Please try again.';
        } else {
            errorMsg += `Error: ${err.name}. Try opening http://localhost:5000 in Chrome/Edge.`;
        }
        updateStatus(errorMsg, 'error');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    webcam.srcObject = null;
    webcamPlaceholder.classList.remove('hidden');
    startCameraBtn.classList.remove('hidden');
    stopCameraBtn.classList.add('hidden');
    captureBtn.classList.add('hidden');
    
    stopAutoDetect();
    clearOverlay(overlay);
    updateStatus('Camera stopped.', '');
}

function captureFrame() {
    if (!stream || isProcessing) return null;
    
    const ctx = processCanvas.getContext('2d');
    processCanvas.width = webcam.videoWidth;
    processCanvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);
    
    return processCanvas.toDataURL('image/jpeg', 0.8);
}

async function analyzeFrame() {
    if (isProcessing) return;
    
    const imageData = captureFrame();
    if (!imageData) return;
    
    isProcessing = true;
    updateStatus('Analyzing...', 'loading');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        handleResult(result, overlay, webcam.videoWidth, webcam.videoHeight);
    } catch (err) {
        console.error('Analysis error:', err);
        updateStatus('Error analyzing image. Please try again.', 'error');
    }
    
    isProcessing = false;
}

// Auto-detect toggle
autoDetectCheckbox.addEventListener('change', (e) => {
    if (e.target.checked && stream) {
        startAutoDetect();
    } else {
        stopAutoDetect();
    }
});

function startAutoDetect() {
    if (autoDetectInterval) return;
    autoDetectInterval = setInterval(analyzeFrame, 300); // ~3 FPS for smoother detection
    updateStatus('Real-time detection active', 'success');
}

function stopAutoDetect() {
    if (autoDetectInterval) {
        clearInterval(autoDetectInterval);
        autoDetectInterval = null;
    }
    autoDetectCheckbox.checked = false;
}

// Event listeners for webcam
startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
captureBtn.addEventListener('click', analyzeFrame);

// File upload functions
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
});

// Drag and drop
const uploadContainer = fileInput.parentElement;
uploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadContainer.classList.add('drag-over');
});

uploadContainer.addEventListener('dragleave', () => {
    uploadContainer.classList.remove('drag-over');
});

uploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadContainer.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    }
});

function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.src = e.target.result;
        uploadedImage.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        analyzeBtn.disabled = false;
        clearOverlay(uploadOverlay);
        updateStatus('Image loaded. Click "Analyze Image" to detect emotions.', 'success');
    };
    reader.readAsDataURL(file);
}

async function analyzeUploadedImage() {
    if (isProcessing || !uploadedImage.src) return;
    
    isProcessing = true;
    analyzeBtn.disabled = true;
    updateStatus('Analyzing...', 'loading');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: uploadedImage.src })
        });
        
        const result = await response.json();
        
        // Get displayed image dimensions for overlay
        const rect = uploadedImage.getBoundingClientRect();
        const scaleX = uploadedImage.naturalWidth / rect.width;
        const scaleY = uploadedImage.naturalHeight / rect.height;
        
        handleResult(result, uploadOverlay, uploadedImage.naturalWidth, uploadedImage.naturalHeight);
    } catch (err) {
        console.error('Analysis error:', err);
        updateStatus('Error analyzing image. Please try again.', 'error');
    }
    
    isProcessing = false;
    analyzeBtn.disabled = false;
}

analyzeBtn.addEventListener('click', analyzeUploadedImage);

// Result handling
function handleResult(result, overlayCanvas, width, height) {
    if (!result.success) {
        if (!isWebcamMode) {
            updateStatus(result.message, 'error');
            mainEmotion.classList.add('hidden');
            faceCount.classList.add('hidden');
            emotionBars.innerHTML = '<p class="text-gray-500 text-center">No emotions detected</p>';
        }
        clearOverlay(overlayCanvas);
        return;
    }
    
    // Draw face boxes on overlay (for both modes)
    drawFaceBoxes(overlayCanvas, result.faces, width, height, isWebcamMode);
    
    // Only update the side panel for upload mode
    if (!isWebcamMode) {
        updateStatus(result.message, 'success');
        faceCount.classList.remove('hidden');
        faceCountNum.textContent = result.faces.length;
        
        if (result.faces.length > 0) {
            const mainPrediction = result.faces[0].prediction;
            displayMainEmotion(mainPrediction);
            displayEmotionBars(mainPrediction.all_emotions);
        }
    }
}

function displayMainEmotion(prediction) {
    mainEmotion.classList.remove('hidden');
    mainEmotion.classList.add('fade-in');
    
    emotionEmoji.textContent = prediction.emoji;
    emotionLabel.textContent = prediction.emotion;
    emotionConfidence.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
}

function displayEmotionBars(emotions) {
    emotionBars.innerHTML = emotions.map((item, index) => `
        <div class="emotion-bar-container ${emotionColors[item.emotion]} ${index === 0 ? 'top-emotion' : ''} fade-in" style="animation-delay: ${index * 0.05}s">
            <span class="emotion-emoji">${item.emoji}</span>
            <span class="emotion-name text-gray-300">${item.emotion}</span>
            <div class="emotion-bar-wrapper">
                <div class="emotion-bar progress-bar" style="width: ${(item.confidence * 100).toFixed(1)}%"></div>
            </div>
            <span class="emotion-percent text-gray-400">${(item.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

function drawFaceBoxes(canvasElement, faces, imgWidth, imgHeight, showDetailedOverlay = false) {
    const ctx = canvasElement.getContext('2d');
    
    // Set canvas size to match container
    const container = canvasElement.parentElement;
    canvasElement.width = container.offsetWidth;
    canvasElement.height = container.offsetHeight;
    
    // Calculate scale
    const scaleX = canvasElement.width / imgWidth;
    const scaleY = canvasElement.height / imgHeight;
    
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    faces.forEach(face => {
        const bbox = face.bbox;
        const prediction = face.prediction;
        
        // Scale coordinates
        const x = bbox.x * scaleX;
        const y = bbox.y * scaleY;
        const w = bbox.w * scaleX;
        const h = bbox.h * scaleY;
        
        // Draw box
        ctx.strokeStyle = '#9333ea';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        
        // Draw label background
        const label = `${prediction.emoji} ${prediction.emotion} ${(prediction.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 14px sans-serif';
        const textWidth = ctx.measureText(label).width;
        
        const gradient = ctx.createLinearGradient(x, y - 25, x + textWidth + 16, y - 25);
        gradient.addColorStop(0, '#9333ea');
        gradient.addColorStop(1, '#db2777');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y - 28, textWidth + 16, 24);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 8, y - 10);
        
        // For webcam mode, show top 3 emotions below the face box
        if (showDetailedOverlay && prediction.all_emotions) {
            const top3 = prediction.all_emotions.slice(0, 3);
            const barWidth = Math.min(w, 150);
            const barHeight = 16;
            const barY = y + h + 8;
            
            top3.forEach((emo, i) => {
                const currentBarY = barY + (i * (barHeight + 4));
                
                // Background bar
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(x, currentBarY, barWidth, barHeight);
                
                // Filled bar based on confidence
                const fillWidth = barWidth * emo.confidence;
                const barGradient = ctx.createLinearGradient(x, currentBarY, x + fillWidth, currentBarY);
                barGradient.addColorStop(0, '#9333ea');
                barGradient.addColorStop(1, '#db2777');
                ctx.fillStyle = barGradient;
                ctx.fillRect(x, currentBarY, fillWidth, barHeight);
                
                // Label
                ctx.fillStyle = 'white';
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(`${emo.emoji} ${emo.emotion} ${(emo.confidence * 100).toFixed(0)}%`, x + 4, currentBarY + 12);
            });
        }
    });
}

function clearOverlay(canvasElement) {
    const ctx = canvasElement.getContext('2d');
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function updateStatus(message, type) {
    statusBar.textContent = message;
    statusBar.className = 'mb-4 p-3 bg-gray-700 rounded-lg text-center';
    
    if (type === 'success') {
        statusBar.classList.add('status-success');
    } else if (type === 'error') {
        statusBar.classList.add('status-error');
    } else if (type === 'loading') {
        statusBar.innerHTML = `<span class="spinner mr-2"></span>${message}`;
        statusBar.classList.add('status-loading');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});
