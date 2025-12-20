// Denoising Autoencoder JavaScript

let canvas, ctx;
let isDrawing = false;
let selectedImageFile = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize canvas
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    // Set canvas background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Canvas event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // File input
    const imageFileInput = document.getElementById('imageFileInput');
    imageFileInput.addEventListener('change', handleImageSelect);
    
    // Noise sliders
    document.getElementById('noiseSlider').addEventListener('input', function(e) {
        document.getElementById('noiseValue').textContent = e.target.value;
    });
    
    document.getElementById('textNoiseSlider').addEventListener('input', function(e) {
        document.getElementById('textNoiseValue').textContent = e.target.value;
    });
    
    // Check model status
    checkModelStatus();
});

// Mode switching
function switchMode(mode) {
    const imageTab = document.getElementById('imageTab');
    const textTab = document.getElementById('textTab');
    const imageMode = document.getElementById('imageMode');
    const textMode = document.getElementById('textMode');
    
    if (mode === 'image') {
        imageTab.classList.add('active');
        textTab.classList.remove('active');
        imageMode.classList.remove('hidden');
        textMode.classList.add('hidden');
    } else {
        textTab.classList.add('active');
        imageTab.classList.remove('active');
        textMode.classList.remove('hidden');
        imageMode.classList.add('hidden');
    }
}

// Canvas drawing
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    ctx.lineTo(x, y);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (touch.clientX - rect.left) * scaleX;
    const y = (touch.clientY - rect.top) * scaleY;
    
    if (e.type === 'touchstart') {
        isDrawing = true;
        ctx.beginPath();
        ctx.moveTo(x, y);
    } else if (e.type === 'touchmove' && isDrawing) {
        ctx.lineTo(x, y);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
    }
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Clear the selected image if it was from canvas
    if (selectedImageFile && typeof selectedImageFile === 'string') {
        selectedImageFile = null;
    }
}

function useCanvasImage() {
    selectedImageFile = canvas.toDataURL('image/png');
    showToast('Drawing ready! Click "Denoise Image" to process.', 'success');
}

function isCanvasBlank() {
    const blank = document.createElement('canvas');
    blank.width = canvas.width;
    blank.height = canvas.height;
    const blankCtx = blank.getContext('2d');
    blankCtx.fillStyle = 'white';
    blankCtx.fillRect(0, 0, blank.width, blank.height);
    
    return canvas.toDataURL() === blank.toDataURL();
}

// File handling
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        selectedImageFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(event) {
            const dropZone = document.getElementById('imageDropZone');
            dropZone.innerHTML = `
                <div class="relative">
                    <img src="${event.target.result}" class="max-w-full max-h-48 mx-auto rounded-lg mb-3">
                    <button onclick="clearUpload()" class="absolute top-2 right-2 px-3 py-1 bg-red-500/80 hover:bg-red-500 text-white rounded-lg text-sm">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <p class="text-sm text-green-300 mt-2"><i class="fas fa-check-circle mr-2"></i>Image loaded successfully</p>
            `;
        };
        reader.readAsDataURL(file);
        
        showToast('Image loaded! Click "Denoise Image" to process.');
    }
}

function clearUpload() {
    selectedImageFile = null;
    document.getElementById('imageFileInput').value = '';
    
    // Reset drop zone
    const dropZone = document.getElementById('imageDropZone');
    dropZone.innerHTML = `
        <i class="fas fa-cloud-upload-alt text-5xl text-purple-400 mb-3"></i>
        <p class="text-lg mb-2">Upload an image</p>
        <label class="inline-block px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg cursor-pointer transition-all">
            <i class="fas fa-folder-open mr-2"></i>Browse Files
            <input type="file" id="imageFileInput" accept="image/*" class="hidden">
        </label>
    `;
    
    // Re-attach event listener
    const imageFileInput = document.getElementById('imageFileInput');
    imageFileInput.addEventListener('change', handleImageSelect);
    
    showToast('Upload cleared', 'error');
}

// Image denoising
async function denoiseImage() {
    // Auto-use canvas if something is drawn and no file is selected
    if (!selectedImageFile && !isCanvasBlank()) {
        useCanvasImage();
    }
    
    if (!selectedImageFile) {
        showToast('Please draw a digit or upload an image first!', 'error');
        return;
    }
    
    const btn = document.getElementById('imageDenoiseBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
    btn.disabled = true;
    
    try {
        const formData = new FormData();
        const noiseFactor = document.getElementById('noiseSlider').value;
        
        if (typeof selectedImageFile === 'string') {
            // Canvas drawing (base64)
            const response = await fetch('/denoise-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: selectedImageFile,
                    noise_factor: noiseFactor
                })
            });
            
            const data = await response.json();
            if (response.ok && data.success) {
                displayImageResults(data);
            } else {
                showToast(data.error || 'Denoising failed', 'error');
            }
        } else {
            // File upload
            formData.append('image', selectedImageFile);
            formData.append('noise_factor', noiseFactor);
            
            const response = await fetch('/denoise-image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            if (response.ok && data.success) {
                displayImageResults(data);
            } else {
                showToast(data.error || 'Denoising failed', 'error');
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function displayImageResults(data) {
    // Show images
    const originalImg = document.getElementById('originalImage');
    const noisyImg = document.getElementById('noisyImage');
    const denoisedImg = document.getElementById('denoisedImage');
    
    originalImg.src = 'data:image/png;base64,' + data.original;
    noisyImg.src = 'data:image/png;base64,' + data.noisy;
    denoisedImg.src = 'data:image/png;base64,' + data.denoised;
    
    originalImg.style.display = 'block';
    noisyImg.style.display = 'block';
    denoisedImg.style.display = 'block';
    
    document.getElementById('originalPlaceholder').style.display = 'none';
    document.getElementById('noisyPlaceholder').style.display = 'none';
    document.getElementById('denoisedPlaceholder').style.display = 'none';
    
    // Show metrics
    document.getElementById('mseNoisy').textContent = data.metrics.mse_noisy.toFixed(4);
    document.getElementById('mseDenoised').textContent = data.metrics.mse_denoised.toFixed(4);
    document.getElementById('improvement').textContent = data.metrics.improvement.toFixed(1) + '%';
    
    showToast('Image denoised successfully!');
}

// Text denoising
function setTextExample(text) {
    document.getElementById('textInput').value = text;
}

async function denoiseText() {
    const text = document.getElementById('textInput').value.trim();
    
    if (!text) {
        showToast('Please enter some text first!', 'error');
        return;
    }
    
    if (text.length > 50) {
        showToast('Text must be 50 characters or less!', 'error');
        return;
    }
    
    const btn = document.getElementById('textDenoiseBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
    btn.disabled = true;
    
    try {
        const noiseLevel = document.getElementById('textNoiseSlider').value;
        
        const response = await fetch('/denoise-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                noise_level: parseFloat(noiseLevel)
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            displayTextResults(data);
        } else {
            showToast(data.error || 'Denoising failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function displayTextResults(data) {
    document.getElementById('originalText').textContent = data.original;
    document.getElementById('noisyText').textContent = data.noisy;
    document.getElementById('denoisedText').textContent = data.denoised;
    
    document.getElementById('textAccuracy').textContent = data.metrics.accuracy.toFixed(1) + '%';
    document.getElementById('textNoiseLevel').textContent = (data.metrics.noise_level * 100).toFixed(0) + '%';
    
    showToast('Text denoised successfully!');
}

// Model status
async function checkModelStatus() {
    try {
        const response = await fetch('/model-status');
        const data = await response.json();
        
        const imageStatus = document.getElementById('imageModelStatus');
        const textStatus = document.getElementById('textModelStatus');
        
        if (data.image_model) {
            imageStatus.querySelector('span').innerHTML = 'Image Model: <span class="font-bold text-green-400">Ready</span>';
        } else {
            imageStatus.querySelector('span').innerHTML = 'Image Model: <span class="font-bold text-yellow-400">Training...</span>';
        }
        
        if (data.text_model) {
            textStatus.querySelector('span').innerHTML = 'Text Model: <span class="font-bold text-green-400">Ready</span>';
        } else {
            textStatus.querySelector('span').innerHTML = 'Text Model: <span class="font-bold text-yellow-400">Training...</span>';
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

// Toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    toastMessage.textContent = message;
    
    if (type === 'error') {
        toast.classList.remove('bg-green-500');
        toast.classList.add('bg-red-500');
    } else {
        toast.classList.remove('bg-red-500');
        toast.classList.add('bg-green-500');
    }
    
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}
