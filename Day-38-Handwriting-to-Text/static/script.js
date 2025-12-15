// Handwriting to Text Generator JavaScript

let canvas, ctx;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let selectedFile = null;

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
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // File input
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Load stats
    loadStats();
});

// Tab switching
function switchTab(tab) {
    const canvasTab = document.getElementById('canvasTab');
    const uploadTab = document.getElementById('uploadTab');
    const canvasSection = document.getElementById('canvasSection');
    const uploadSection = document.getElementById('uploadSection');
    
    if (tab === 'canvas') {
        canvasTab.classList.add('active');
        uploadTab.classList.remove('active');
        canvasSection.classList.remove('hidden');
        uploadSection.classList.add('hidden');
    } else {
        uploadTab.classList.add('active');
        canvasTab.classList.remove('active');
        uploadSection.classList.remove('hidden');
        canvasSection.classList.add('hidden');
    }
}

// Canvas drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    const penSize = document.getElementById('penSize').value;
    const penColor = document.getElementById('penColor').value;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.strokeStyle = penColor;
    ctx.lineWidth = penSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    if (e.type === 'touchstart') {
        isDrawing = true;
        [lastX, lastY] = [x, y];
    } else if (e.type === 'touchmove' && isDrawing) {
        const penSize = document.getElementById('penSize').value;
        const penColor = document.getElementById('penColor').value;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = penColor;
        ctx.lineWidth = penSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
    }
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// File handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayImagePreview(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        displayImagePreview(file);
    } else {
        showToast('Please drop an image file', 'error');
    }
}

function displayImagePreview(file) {
    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const img = document.getElementById('previewImg');
        img.src = e.target.result;
        document.getElementById('imagePreview').classList.remove('hidden');
    };
    
    reader.readAsDataURL(file);
}

// Recognition functions
async function recognizeCanvas() {
    const btn = document.getElementById('canvasBtn');
    const originalText = btn.innerHTML;
    
    // Show loading
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
    btn.disabled = true;
    btn.classList.add('loading');
    
    try {
        // Get canvas image data
        const imageData = canvas.toDataURL('image/png');
        
        const response = await fetch('/recognize-canvas', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            showToast(data.error || 'Recognition failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
        btn.classList.remove('loading');
    }
}

async function recognizeUpload() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }
    
    const btn = document.getElementById('uploadBtn');
    const originalText = btn.innerHTML;
    
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
    btn.disabled = true;
    btn.classList.add('loading');
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        const response = await fetch('/recognize-upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            showToast(data.error || 'Recognition failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
        btn.classList.remove('loading');
    }
}

// Display results
function displayResult(data) {
    // Update stats
    document.getElementById('wordCount').textContent = data.word_count;
    document.getElementById('lineCount').textContent = data.line_count;
    document.getElementById('charCount').textContent = data.char_count;
    document.getElementById('confidence').textContent = data.confidence + '%';
    document.getElementById('processingTime').textContent = data.processing_time + 's';
    
    // Update text with typing effect
    const textElement = document.getElementById('recognizedText');
    textElement.textContent = '';
    
    let i = 0;
    const text = data.text;
    const typingSpeed = 20; // ms per character
    
    function typeWriter() {
        if (i < text.length) {
            textElement.textContent += text.charAt(i);
            i++;
            setTimeout(typeWriter, typingSpeed);
        }
    }
    
    typeWriter();
    
    // Scroll to result on mobile
    if (window.innerWidth < 1024) {
        setTimeout(() => {
            document.getElementById('resultSection').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);
    }
}

// Copy and download functions
function copyText() {
    const text = document.getElementById('recognizedText').textContent;
    
    if (!text) {
        showToast('No text to copy', 'error');
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        showToast('Text copied to clipboard!');
    }).catch(err => {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('Text copied to clipboard!');
    });
}

function downloadText() {
    const text = document.getElementById('recognizedText').textContent;
    
    if (!text) {
        showToast('No text to download', 'error');
        return;
    }
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `handwriting-text-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Text downloaded!');
}

// Example drawings
function drawExample(type) {
    clearCanvas();
    
    ctx.fillStyle = 'black';
    ctx.font = '24px cursive';
    
    switch(type) {
        case 'note':
            ctx.fillText('Remember to buy milk!', 50, 100);
            ctx.fillText('Meeting at 3 PM', 50, 150);
            break;
        case 'signature':
            ctx.font = '32px cursive';
            ctx.fillText('John Doe', 200, 150);
            break;
        case 'list':
            ctx.font = '20px cursive';
            ctx.fillText('1. Bread', 50, 80);
            ctx.fillText('2. Eggs', 50, 120);
            ctx.fillText('3. Butter', 50, 160);
            ctx.fillText('4. Milk', 50, 200);
            break;
        case 'quote':
            ctx.font = '18px cursive';
            ctx.fillText('Success is not final,', 50, 100);
            ctx.fillText('failure is not fatal:', 50, 140);
            ctx.fillText('it is the courage to continue', 50, 180);
            ctx.fillText('that counts.', 50, 220);
            break;
    }
}

// Load model stats
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        const modelName = data.model.split('/')[1];
        document.getElementById('modelStatus').textContent = `${modelName} Ready`;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('modelStatus').textContent = 'Ready';
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
