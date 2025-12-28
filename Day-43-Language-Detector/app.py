"""
Day 43: Language Detector (Multilingual LLM)
Detect language using LLM zero-shot classification
"""

import re
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
# Using XLM-RoBERTa based model for multilingual zero-shot classification
MODEL_NAME = "facebook/bart-large-mnli"  # Good for zero-shot classification

# Supported languages for detection
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Dutch", "Russian", "Chinese", "Japanese", "Korean", "Arabic",
    "Hindi", "Turkish", "Polish", "Swedish", "Norwegian", "Danish",
    "Finnish", "Greek", "Hebrew", "Thai", "Vietnamese", "Indonesian",
    "Malay", "Filipino", "Czech", "Romanian", "Hungarian", "Ukrainian"
]

# Language metadata (ISO codes and native names)
LANGUAGE_INFO = {
    "English": {"code": "en", "native": "English", "flag": "🇬🇧"},
    "Spanish": {"code": "es", "native": "Español", "flag": "🇪🇸"},
    "French": {"code": "fr", "native": "Français", "flag": "🇫🇷"},
    "German": {"code": "de", "native": "Deutsch", "flag": "🇩🇪"},
    "Italian": {"code": "it", "native": "Italiano", "flag": "🇮🇹"},
    "Portuguese": {"code": "pt", "native": "Português", "flag": "🇵🇹"},
    "Dutch": {"code": "nl", "native": "Nederlands", "flag": "🇳🇱"},
    "Russian": {"code": "ru", "native": "Русский", "flag": "🇷🇺"},
    "Chinese": {"code": "zh", "native": "中文", "flag": "🇨🇳"},
    "Japanese": {"code": "ja", "native": "日本語", "flag": "🇯🇵"},
    "Korean": {"code": "ko", "native": "한국어", "flag": "🇰🇷"},
    "Arabic": {"code": "ar", "native": "العربية", "flag": "🇸🇦"},
    "Hindi": {"code": "hi", "native": "हिन्दी", "flag": "🇮🇳"},
    "Turkish": {"code": "tr", "native": "Türkçe", "flag": "🇹🇷"},
    "Polish": {"code": "pl", "native": "Polski", "flag": "🇵🇱"},
    "Swedish": {"code": "sv", "native": "Svenska", "flag": "🇸🇪"},
    "Norwegian": {"code": "no", "native": "Norsk", "flag": "🇳🇴"},
    "Danish": {"code": "da", "native": "Dansk", "flag": "🇩🇰"},
    "Finnish": {"code": "fi", "native": "Suomi", "flag": "🇫🇮"},
    "Greek": {"code": "el", "native": "Ελληνικά", "flag": "🇬🇷"},
    "Hebrew": {"code": "he", "native": "עברית", "flag": "🇮🇱"},
    "Thai": {"code": "th", "native": "ไทย", "flag": "🇹🇭"},
    "Vietnamese": {"code": "vi", "native": "Tiếng Việt", "flag": "🇻🇳"},
    "Indonesian": {"code": "id", "native": "Bahasa Indonesia", "flag": "🇮🇩"},
    "Malay": {"code": "ms", "native": "Bahasa Melayu", "flag": "🇲🇾"},
    "Filipino": {"code": "tl", "native": "Filipino", "flag": "🇵🇭"},
    "Czech": {"code": "cs", "native": "Čeština", "flag": "🇨🇿"},
    "Romanian": {"code": "ro", "native": "Română", "flag": "🇷🇴"},
    "Hungarian": {"code": "hu", "native": "Magyar", "flag": "🇭🇺"},
    "Ukrainian": {"code": "uk", "native": "Українська", "flag": "🇺🇦"},
}

# Global classifier
classifier = None


def load_model():
    """Load zero-shot classification model"""
    global classifier
    
    print(f"📦 Loading language detection model: {MODEL_NAME}...")
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1  # CPU
    )
    print("✅ Model loaded successfully!")
    return classifier


def detect_language(text, top_k=5):
    """
    Detect language using zero-shot classification
    
    Args:
        text: Input text to analyze
        top_k: Number of top languages to return
    
    Returns:
        List of detected languages with confidence scores
    """
    global classifier
    
    if classifier is None:
        load_model()
    
    # Clean text
    text = text.strip()
    if not text:
        return []
    
    # Create hypothesis template for zero-shot
    hypothesis_template = "This text is written in {}."
    
    try:
        # Run classification
        result = classifier(
            text,
            LANGUAGES,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        
        # Format results
        detections = []
        for i in range(min(top_k, len(result['labels']))):
            lang = result['labels'][i]
            score = result['scores'][i]
            info = LANGUAGE_INFO.get(lang, {"code": "??", "native": lang, "flag": "🏳️"})
            
            detections.append({
                'language': lang,
                'confidence': round(score * 100, 2),
                'code': info['code'],
                'native_name': info['native'],
                'flag': info['flag']
            })
        
        return detections
        
    except Exception as e:
        print(f"Error detecting language: {e}")
        return []


def detect_batch(texts, top_k=3):
    """Detect languages for multiple texts"""
    results = []
    for text in texts:
        detection = detect_language(text, top_k)
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'detections': detection
        })
    return results


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Detect language endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        top_k = int(data.get('top_k', 5))
        
        if not text:
            return jsonify({'success': False, 'error': 'Please provide text to analyze'}), 400
        
        if len(text) < 3:
            return jsonify({'success': False, 'error': 'Text too short. Please provide at least 3 characters.'}), 400
        
        # Detect language
        detections = detect_language(text, top_k)
        
        if not detections:
            return jsonify({'success': False, 'error': 'Could not detect language'}), 500
        
        # Primary detection
        primary = detections[0]
        
        return jsonify({
            'success': True,
            'primary': primary,
            'all_detections': detections,
            'text_length': len(text),
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect-batch', methods=['POST'])
def detect_batch_endpoint():
    """Detect languages for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        top_k = int(data.get('top_k', 3))
        
        if not texts:
            return jsonify({'success': False, 'error': 'Please provide texts to analyze'}), 400
        
        # Filter empty texts
        texts = [t.strip() for t in texts if t.strip()]
        
        if not texts:
            return jsonify({'success': False, 'error': 'No valid texts provided'}), 400
        
        # Detect languages
        results = detect_batch(texts, top_k)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_texts': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/sample-texts', methods=['GET'])
def sample_texts():
    """Get sample texts in different languages"""
    samples = [
        {"text": "Hello, how are you today? I hope you're having a wonderful day!", "expected": "English"},
        {"text": "Bonjour, comment allez-vous? J'espère que vous passez une excellente journée!", "expected": "French"},
        {"text": "Hola, ¿cómo estás hoy? ¡Espero que tengas un día maravilloso!", "expected": "Spanish"},
        {"text": "Guten Tag, wie geht es Ihnen? Ich hoffe, Sie haben einen wunderbaren Tag!", "expected": "German"},
        {"text": "Ciao, come stai oggi? Spero che tu stia passando una giornata meravigliosa!", "expected": "Italian"},
        {"text": "Olá, como você está hoje? Espero que você esteja tendo um dia maravilhoso!", "expected": "Portuguese"},
        {"text": "Привет, как у тебя дела сегодня? Надеюсь, у тебя прекрасный день!", "expected": "Russian"},
        {"text": "你好，今天过得怎么样？希望你今天过得愉快！", "expected": "Chinese"},
        {"text": "こんにちは、今日の調子はいかがですか？素敵な一日をお過ごしください！", "expected": "Japanese"},
        {"text": "안녕하세요, 오늘 기분이 어떠세요? 좋은 하루 보내세요!", "expected": "Korean"},
        {"text": "مرحبا، كيف حالك اليوم؟ أتمنى لك يوما رائعا!", "expected": "Arabic"},
        {"text": "नमस्ते, आज आप कैसे हैं? मुझे आशा है कि आपका दिन शानदार हो!", "expected": "Hindi"},
    ]
    return jsonify({'samples': samples})


@app.route('/languages', methods=['GET'])
def get_languages():
    """Get all supported languages"""
    languages = []
    for lang in LANGUAGES:
        info = LANGUAGE_INFO.get(lang, {"code": "??", "native": lang, "flag": "🏳️"})
        languages.append({
            'name': lang,
            'code': info['code'],
            'native': info['native'],
            'flag': info['flag']
        })
    return jsonify({'languages': languages, 'total': len(languages)})


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    return jsonify({
        'model_loaded': classifier is not None,
        'model_name': MODEL_NAME,
        'supported_languages': len(LANGUAGES)
    })


# ===============================
# INITIALIZATION
# ===============================

print("🌍 Language Detector - Day 43")
print("=" * 35)

try:
    load_model()
except Exception as e:
    print(f"⚠️ Model will be loaded on first request: {e}")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
