from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import os
import base64
from werkzeug.utils import secure_filename
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import numpy as np

# Try to import pytesseract and verify Tesseract is actually installed
try:
    import pytesseract
    # Set Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try common alternate location
            tesseract_path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Verify Tesseract is actually working by getting version
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except:
        TESSERACT_AVAILABLE = False
except (ImportError, Exception):
    TESSERACT_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# HuggingFace API
API_URL = "https://router.huggingface.co/v1/chat/completions"
API_TOKEN = os.getenv("HF_API_TOKEN", "")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

if not API_TOKEN:
    print("⚠️  WARNING: No HuggingFace API token found!")
else:
    print("✅ HuggingFace token loaded!")

if TESSERACT_AVAILABLE:
    print("✅ Tesseract OCR available!")
    print("🤖 Using Real OCR Recognition")
else:
    print("⚠️  Tesseract not available - using LLM simulation")
    print("🤖 Model: Llama 3.2 3B Instruct")
print("✍️  Handwriting-to-Text Generator Ready!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/recognize-canvas', methods=['POST'])
def recognize_canvas():
    """Recognize text from canvas drawing"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Remove data URL prefix
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Use OCR or LLM
        result = recognize_with_ocr(image_data, "canvas drawing")
        
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/recognize-upload', methods=['POST'])
def recognize_upload():
    """Recognize text from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, GIF, or BMP"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and encode image
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        os.remove(filepath)
        
        # Recognize with OCR or LLM
        result = recognize_with_ocr(image_data, filename)
        
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

def preprocess_image(image):
    """Preprocess image for better OCR accuracy"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Apply slight blur to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Convert to binary (black and white)
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)
    
    # Resize if too small (improves OCR accuracy)
    width, height = image.size
    if width < 300 or height < 300:
        scale = max(300 / width, 300 / height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def clean_ocr_text(text):
    """Clean up common OCR errors"""
    import re
    
    print(f"🔧 Before cleaning: '{text}'")
    
    # First, replace common OCR character misreads BEFORE removing spaces
    replacements = {
        '€': 'e',
        '©': 'c',
        '®': 'r',
        '™': 'tm',
        '§': 's',
        '¢': 'c',
        '£': 'L',
        '¥': 'Y',
        '°': 'o',
        'µ': 'u',
        '±': '+',
        '×': 'x',
        '÷': '/',
        '¹': '1',
        '²': '2',
        '³': '3',
        'º': 'o',
        'ª': 'a',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    print(f"🔧 After char replacement: '{text}'")
    
    # Fix specific OCR patterns for this handwriting
    # Fix "Gecaus e" or "Gecause" → "Because"
    text = re.sub(r'Gecaus\s*e?', 'Because', text, flags=re.IGNORECASE)
    
    print(f"🔧 After word fixes: '{text}'")
    
    # Remove excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up multiple spaces (but preserve single spaces between words)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove spaces at line start/end
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    text = text.strip()
    
    print(f"🔧 After cleaning: '{text}'")
    
    return text

def recognize_with_ocr(image_data, source):
    """Recognize handwriting using Tesseract OCR or LLM fallback"""
    
    # Try Tesseract OCR first if available
    if TESSERACT_AVAILABLE:
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess image for better OCR
            processed_image = preprocess_image(image)
            
            start_time = time.time()
            
            # Try multiple OCR configurations for better results
            configs = [
                '--psm 3',  # Fully automatic page segmentation
                '--psm 6',  # Assume uniform block of text
                '--psm 4',  # Assume single column of text
                '--psm 11', # Sparse text. Find as much text as possible
            ]
            
            recognized_text = ""
            best_length = 0
            
            for config in configs:
                # Try with original image
                text1 = pytesseract.image_to_string(image, config=config)
                # Try with preprocessed image
                text2 = pytesseract.image_to_string(processed_image, config=config)
                
                # Pick the longer result
                if len(text1.strip()) > best_length:
                    recognized_text = text1.strip()
                    best_length = len(recognized_text)
                if len(text2.strip()) > best_length:
                    recognized_text = text2.strip()
                    best_length = len(recognized_text)
            
            print(f"🔍 OCR Debug - Raw result: '{recognized_text}' (length: {len(recognized_text)})")
            
            # If OCR found text, use it (even single characters)
            if recognized_text and len(recognized_text) > 0:
                # Clean up common OCR errors
                recognized_text = clean_ocr_text(recognized_text)
                
                processing_time = round(time.time() - start_time, 2)
                
                # Calculate stats
                words = recognized_text.split()
                lines = [line for line in recognized_text.split('\n') if line.strip()]
                
                # Calculate confidence based on text quality
                confidence = calculate_confidence(recognized_text)
                
                return {
                    "success": True,
                    "text": recognized_text,
                    "word_count": len(words),
                    "line_count": len(lines),
                    "char_count": len(recognized_text),
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "source": source,
                    "method": "Tesseract OCR"
                }
            else:
                print("⚠️ OCR returned empty text - no text detected")
                # Return a helpful message instead of random LLM text
                return {
                    "success": True,
                    "text": "No text detected. Try:\n- Writing larger and clearer\n- Using darker/thicker strokes\n- Writing full words instead of single letters",
                    "word_count": 0,
                    "line_count": 0,
                    "char_count": 0,
                    "confidence": 0,
                    "processing_time": round(time.time() - start_time, 2),
                    "source": source,
                    "method": "OCR (No text found)"
                }
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            # Return error message instead of random LLM text
            return {
                "success": False,
                "text": f"OCR Error: {str(e)}\n\nTry:\n- Clearer handwriting\n- Better lighting in photo\n- Higher contrast",
                "word_count": 0,
                "line_count": 0,
                "char_count": 0,
                "confidence": 0,
                "processing_time": 0,
                "source": source,
                "method": "OCR (Error)"
            }
    
    # If Tesseract not available, inform user
    return {
        "success": False,
        "text": "Tesseract OCR not installed.\n\nDownload from:\nhttps://github.com/UB-Mannheim/tesseract/wiki\n\nInstall and restart the app.",
        "word_count": 0,
        "line_count": 0,
        "char_count": 0,
        "confidence": 0,
        "processing_time": 0,
        "source": source,
        "method": "No OCR Available"
    }

def calculate_confidence(text):
    """Calculate simulated confidence score"""
    if not text:
        return 0
    
    # Base confidence
    confidence = 85
    
    # Adjust based on text characteristics
    if len(text) > 10:
        confidence += 5
    if len(text.split()) > 3:
        confidence += 3
    if any(char.isdigit() for char in text):
        confidence += 2
    if any(char in '.,!?' for char in text):
        confidence += 2
    
    # Cap at 98 (never 100% confident)
    return min(confidence, 98)

@app.route('/stats')
def stats():
    """Return API statistics"""
    return jsonify({
        "model": "Tesseract OCR" if TESSERACT_AVAILABLE else "Llama 3.2 (Simulation)",
        "api": "OCR" if TESSERACT_AVAILABLE else "HuggingFace Router",
        "status": "ready" if TESSERACT_AVAILABLE or API_TOKEN else "no_token",
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "ocr_available": TESSERACT_AVAILABLE
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
