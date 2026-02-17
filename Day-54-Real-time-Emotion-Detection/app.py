import os
import cv2
import numpy as np
import base64
import urllib.request
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models

app = Flask(__name__)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = ['😠', '🤢', '😨', '😊', '😢', '😲', '😐']

# Global model variable
emotion_model = None
face_cascade = None

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)


def build_emotion_model():
    """
    Build mini-XCEPTION model for emotion detection.
    Architecture matches pre-trained weights from FER2013.
    """
    from tensorflow.keras.layers import (Conv2D, SeparableConv2D, MaxPooling2D,
                                         GlobalAveragePooling2D, BatchNormalization,
                                         Activation, Add, Input)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    
    # Mini-XCEPTION architecture
    img_input = Input(shape=(64, 64, 1))
    
    # Block 1
    x = Conv2D(8, (3, 3), strides=(1, 1), use_bias=False, padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    
    # Module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    
    # Module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    
    # Module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    
    # Output
    x = Conv2D(7, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    model = Model(img_input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def download_pretrained_weights():
    """Download pre-trained FER weights."""
    weights_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    weights_path = "models/fer2013_mini_XCEPTION.hdf5"
    
    if not os.path.exists(weights_path):
        print("📥 Downloading pre-trained emotion model weights...")
        print("   This may take a minute...")
        try:
            urllib.request.urlretrieve(weights_url, weights_path)
            print("✅ Weights downloaded successfully!")
            return weights_path
        except Exception as e:
            print(f"⚠️ Could not download weights: {e}")
            return None
    return weights_path


def load_or_create_model():
    """Load pre-trained model or create new one."""
    global emotion_model
    
    # Try to load pre-trained weights
    weights_path = download_pretrained_weights()
    
    if weights_path and os.path.exists(weights_path):
        print("✅ Loading pre-trained emotion model...")
        try:
            emotion_model = tf.keras.models.load_model(weights_path, compile=False)
            emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("✅ Pre-trained model loaded successfully!")
            return emotion_model
        except Exception as e:
            print(f"⚠️ Error loading pre-trained model: {e}")
    
    # Fallback: create new model (won't work well without training)
    print("🔧 Creating new model (demo mode - results will be inaccurate)...")
    emotion_model = build_emotion_model()
    return emotion_model


def initialize_face_detector():
    """Initialize OpenCV face detector."""
    global face_cascade
    
    # Try to load Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("⚠️ Warning: Could not load face cascade classifier")
        return False
    
    print("✅ Face detector initialized")
    return True


def preprocess_face(face_img):
    """
    Preprocess face image for emotion model.
    - Convert to grayscale if needed
    - Apply histogram equalization for better contrast
    - Resize to 64x64 (matches pre-trained model)
    - Normalize pixel values
    """
    # Convert to grayscale if RGB
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This significantly improves detection in varying lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_enhanced = clahe.apply(face_gray)

    # Resize to 64x64 (mini-XCEPTION input size)
    face_resized = cv2.resize(face_enhanced, (64, 64))

    # Normalize to [0, 1]
    face_normalized = face_resized.astype('float32') / 255.0

    # Reshape for model input (batch_size, height, width, channels)
    face_input = face_normalized.reshape(1, 64, 64, 1)

    return face_input


def detect_faces(image):
    """Detect faces in an image using OpenCV Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better face detection in low light
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,  # Slightly lower for better detection
        minSize=(60, 60),  # Larger minimum for better quality faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces


def predict_emotion(face_img):
    """Predict emotion from a face image."""
    preprocessed = preprocess_face(face_img)
    predictions = emotion_model.predict(preprocessed, verbose=0)[0]
    
    # Get top emotion
    emotion_idx = np.argmax(predictions)
    emotion = EMOTIONS[emotion_idx]
    emoji = EMOTION_EMOJIS[emotion_idx]
    confidence = float(predictions[emotion_idx])
    
    # Get all emotions with their scores
    all_emotions = [
        {
            'emotion': EMOTIONS[i],
            'emoji': EMOTION_EMOJIS[i],
            'confidence': float(predictions[i])
        }
        for i in range(len(EMOTIONS))
    ]
    
    # Sort by confidence
    all_emotions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'emotion': emotion,
        'emoji': emoji,
        'confidence': confidence,
        'all_emotions': all_emotions
    }


def process_image(image_data):
    """Process image data and return emotion predictions for all detected faces."""
    # Decode base64 image
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to OpenCV format (BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces = detect_faces(image_cv)
    
    if len(faces) == 0:
        return {
            'success': False,
            'message': 'No face detected. Please ensure your face is visible and well-lit.',
            'faces': []
        }
    
    results = []
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = image_cv[y:y+h, x:x+w]
        
        # Predict emotion
        prediction = predict_emotion(face_img)
        
        results.append({
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'prediction': prediction
        })
    
    return {
        'success': True,
        'message': f'Detected {len(faces)} face(s)',
        'faces': results
    }


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle emotion prediction requests."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        result = process_image(data['image'])
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': emotion_model is not None,
        'face_detector_loaded': face_cascade is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 Real-time Emotion Detection - Day 54")
    print("="*60 + "\n")
    
    # Initialize components
    initialize_face_detector()
    load_or_create_model()
    
    print("\n" + "-"*60)
    print("🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("-"*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
