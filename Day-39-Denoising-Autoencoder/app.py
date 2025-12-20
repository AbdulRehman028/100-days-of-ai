"""
Day 39: Denoising Autoencoder for Text/Images
Train autoencoders to denoise noisy inputs with real-time evaluation
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import string
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models
image_autoencoder = None
text_autoencoder = None
text_vocab = None

# ====================
# IMAGE AUTOENCODER
# ====================

def build_image_autoencoder():
    """Build convolutional autoencoder for image denoising"""
    # Encoder
    encoder_input = layers.Input(shape=(28, 28, 1), name='encoder_input')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='encoded')(x)
    
    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
    
    # Build model
    autoencoder = models.Model(encoder_input, decoder_output, name='image_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder


def add_noise_to_images(images, noise_factor=0.3):
    """Add Gaussian noise to images"""
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images


def train_image_autoencoder():
    """Train image denoising autoencoder on MNIST"""
    global image_autoencoder
    
    print("Loading MNIST dataset...")
    (x_train, _), (x_test, _) = mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Add noise
    print("Adding noise to images...")
    x_train_noisy = add_noise_to_images(x_train, noise_factor=0.5)
    x_test_noisy = add_noise_to_images(x_test, noise_factor=0.5)
    
    # Build and train model
    print("Building model...")
    image_autoencoder = build_image_autoencoder()
    
    print("Training model...")
    history = image_autoencoder.fit(
        x_train_noisy, x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        verbose=1
    )
    
    # Save model
    image_autoencoder.save('models/image_autoencoder.h5')
    print("Model saved successfully!")
    
    return history


def load_image_autoencoder():
    """Load pre-trained image autoencoder"""
    global image_autoencoder
    model_path = 'models/image_autoencoder.h5'
    
    if os.path.exists(model_path):
        print("Loading pre-trained image autoencoder...")
        try:
            image_autoencoder = keras.models.load_model(model_path, compile=False)
            image_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print("Image autoencoder loaded!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Retraining image autoencoder...")
            train_image_autoencoder()
            return True
    else:
        print("Training new image autoencoder...")
        train_image_autoencoder()
        return True


# ====================
# TEXT AUTOENCODER
# ====================

def build_text_vocab():
    """Build vocabulary for text autoencoder"""
    chars = string.ascii_lowercase + string.digits + string.punctuation + ' '
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 reserved for padding
    char_to_idx['<PAD>'] = 0
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def build_text_autoencoder(vocab_size=100, max_len=50):
    """Build LSTM autoencoder for text denoising"""
    # Encoder
    encoder_input = layers.Input(shape=(max_len,), name='encoder_input')
    x = layers.Embedding(vocab_size, 128)(encoder_input)
    x = layers.LSTM(128, return_sequences=True)(x)
    encoded = layers.LSTM(64)(x)
    
    # Decoder
    x = layers.RepeatVector(max_len)(encoded)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    decoder_output = layers.Dense(vocab_size, activation='softmax')(x)
    
    # Build model
    autoencoder = models.Model(encoder_input, decoder_output, name='text_autoencoder')
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return autoencoder


def generate_sample_texts(num_samples=5000):
    """Generate sample texts for training"""
    templates = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world this is a test",
        "Machine learning is awesome",
        "Deep learning models are powerful",
        "Natural language processing",
        "Python programming language",
        "Artificial intelligence research",
        "Data science and analytics",
        "Neural networks learn patterns",
        "Autoencoder removes noise"
    ]
    
    samples = []
    for _ in range(num_samples):
        text = random.choice(templates)
        # Add variations
        if random.random() > 0.5:
            text = text.lower()
        if random.random() > 0.7:
            text = text.upper()
        samples.append(text)
    
    return samples


def text_to_sequence(text, char_to_idx, max_len=50):
    """Convert text to sequence of indices"""
    sequence = [char_to_idx.get(c.lower(), 0) for c in text[:max_len]]
    sequence = sequence + [0] * (max_len - len(sequence))  # Pad
    return np.array(sequence)


def sequence_to_text(sequence, idx_to_char):
    """Convert sequence of indices to text"""
    text = ''.join([idx_to_char.get(idx, '') for idx in sequence if idx != 0])
    return text


def add_noise_to_text(sequences, noise_level=0.2, vocab_size=100):
    """Add noise to text sequences by randomly changing characters"""
    noisy = sequences.copy()
    mask = np.random.random(sequences.shape) < noise_level
    noise = np.random.randint(1, vocab_size, sequences.shape)
    noisy[mask] = noise[mask]
    return noisy


def train_text_autoencoder():
    """Train text denoising autoencoder"""
    global text_autoencoder, text_vocab
    
    print("Building text vocabulary...")
    char_to_idx, idx_to_char = build_text_vocab()
    text_vocab = (char_to_idx, idx_to_char)
    vocab_size = len(char_to_idx)
    
    print("Generating sample texts...")
    texts = generate_sample_texts(5000)
    
    # Convert to sequences
    max_len = 50
    sequences = np.array([text_to_sequence(text, char_to_idx, max_len) for text in texts])
    
    # Add noise
    print("Adding noise to texts...")
    noisy_sequences = add_noise_to_text(sequences, noise_level=0.2, vocab_size=vocab_size)
    
    # Prepare targets (one-hot encoded)
    sequences_expanded = np.expand_dims(sequences, -1)
    
    # Split train/test
    split = int(0.8 * len(sequences))
    x_train, x_test = noisy_sequences[:split], noisy_sequences[split:]
    y_train, y_test = sequences_expanded[:split], sequences_expanded[split:]
    
    # Build and train
    print("Building text autoencoder...")
    text_autoencoder = build_text_autoencoder(vocab_size=vocab_size, max_len=max_len)
    
    print("Training text autoencoder...")
    history = text_autoencoder.fit(
        x_train, y_train,
        epochs=20,
        batch_size=64,
        shuffle=True,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Save model and vocab
    text_autoencoder.save('models/text_autoencoder.h5')
    np.save('models/text_vocab.npy', text_vocab, allow_pickle=True)
    print("Text model saved successfully!")
    
    return history


def load_text_autoencoder():
    """Load pre-trained text autoencoder"""
    global text_autoencoder, text_vocab
    
    model_path = 'models/text_autoencoder.h5'
    vocab_path = 'models/text_vocab.npy'
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("Loading pre-trained text autoencoder...")
        try:
            text_autoencoder = keras.models.load_model(model_path, compile=False)
            text_autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            text_vocab = tuple(np.load(vocab_path, allow_pickle=True))
            print("Text autoencoder loaded!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Retraining text autoencoder...")
            train_text_autoencoder()
            return True
    else:
        print("Training new text autoencoder...")
        train_text_autoencoder()
        return True


# ====================
# FLASK ROUTES
# ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/train-models', methods=['POST'])
def train_models():
    """Train both autoencoders"""
    try:
        model_type = request.json.get('type', 'both')
        
        results = {}
        
        if model_type in ['image', 'both']:
            print("Training image autoencoder...")
            train_image_autoencoder()
            results['image'] = 'success'
        
        if model_type in ['text', 'both']:
            print("Training text autoencoder...")
            train_text_autoencoder()
            results['text'] = 'success'
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully!',
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/denoise-image', methods=['POST'])
def denoise_image():
    """Denoise uploaded image"""
    try:
        if image_autoencoder is None:
            load_image_autoencoder()
        
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('L')
        else:
            # Base64 encoded image
            img_data = request.json.get('image')
            img_data = base64.b64decode(img_data.split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Add noise
        noise_factor = float(request.form.get('noise_factor', 0.3))
        noisy_img = add_noise_to_images(img_array, noise_factor=noise_factor)
        
        # Denoise
        denoised_img = image_autoencoder.predict(noisy_img, verbose=0)
        
        # Convert to base64
        def img_to_base64(img_arr):
            img_arr = (img_arr.squeeze() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_arr, mode='L')
            buffered = io.BytesIO()
            img_pil.save(buffered, format='PNG')
            return base64.b64encode(buffered.getvalue()).decode()
        
        original_b64 = img_to_base64(img_array)
        noisy_b64 = img_to_base64(noisy_img)
        denoised_b64 = img_to_base64(denoised_img)
        
        # Calculate metrics
        mse_noisy = np.mean((img_array - noisy_img) ** 2)
        mse_denoised = np.mean((img_array - denoised_img) ** 2)
        improvement = ((mse_noisy - mse_denoised) / mse_noisy * 100) if mse_noisy > 0 else 0
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'noisy': noisy_b64,
            'denoised': denoised_b64,
            'metrics': {
                'mse_noisy': float(mse_noisy),
                'mse_denoised': float(mse_denoised),
                'improvement': float(improvement)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/denoise-text', methods=['POST'])
def denoise_text():
    """Denoise text input"""
    try:
        if text_autoencoder is None or text_vocab is None:
            load_text_autoencoder()
        
        char_to_idx, idx_to_char = text_vocab
        
        # Get input text
        text = request.json.get('text', '')
        noise_level = float(request.json.get('noise_level', 0.2))
        
        # Convert to sequence
        max_len = 50
        sequence = text_to_sequence(text, char_to_idx, max_len)
        sequence = np.expand_dims(sequence, 0)
        
        # Add noise
        vocab_size = len(char_to_idx)
        noisy_sequence = add_noise_to_text(sequence, noise_level=noise_level, vocab_size=vocab_size)
        
        # Denoise
        predictions = text_autoencoder.predict(noisy_sequence, verbose=0)
        denoised_sequence = np.argmax(predictions, axis=-1)[0]
        
        # Convert back to text
        noisy_text = sequence_to_text(noisy_sequence[0], idx_to_char)
        denoised_text = sequence_to_text(denoised_sequence, idx_to_char)
        
        # Calculate accuracy
        original_chars = [c for c in text.lower()[:max_len]]
        denoised_chars = [c for c in denoised_text[:len(original_chars)]]
        accuracy = sum(1 for a, b in zip(original_chars, denoised_chars) if a == b) / len(original_chars) * 100 if original_chars else 0
        
        return jsonify({
            'success': True,
            'original': text,
            'noisy': noisy_text,
            'denoised': denoised_text,
            'metrics': {
                'accuracy': float(accuracy),
                'noise_level': float(noise_level)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model training status"""
    return jsonify({
        'image_model': os.path.exists('models/image_autoencoder.h5'),
        'text_model': os.path.exists('models/text_autoencoder.h5')
    })


# Load models on startup
try:
    load_image_autoencoder()
    load_text_autoencoder()
except Exception as e:
    print(f"Error loading models: {e}")
    print("Models will be trained on first use.")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
