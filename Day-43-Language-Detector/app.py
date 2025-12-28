import re
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
# Using XLM-RoBERTa fine-tuned for language detection (20 languages, high accuracy)
MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

# Language code to name mapping (model outputs ISO codes)
LANGUAGE_CODES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"
}

# Language metadata (keyed by ISO code for the new model)
LANGUAGE_INFO = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸"},
    "fr": {"name": "French", "native": "Français", "flag": "🇫🇷"},
    "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪"},
    "it": {"name": "Italian", "native": "Italiano", "flag": "🇮🇹"},
    "pt": {"name": "Portuguese", "native": "Português", "flag": "🇵🇹"},
    "nl": {"name": "Dutch", "native": "Nederlands", "flag": "🇳🇱"},
    "ru": {"name": "Russian", "native": "Русский", "flag": "🇷🇺"},
    "zh": {"name": "Chinese", "native": "中文", "flag": "🇨🇳"},
    "ja": {"name": "Japanese", "native": "日本語", "flag": "🇯🇵"},
    "ar": {"name": "Arabic", "native": "العربية", "flag": "🇸🇦"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳"},
    "tr": {"name": "Turkish", "native": "Türkçe", "flag": "🇹🇷"},
    "pl": {"name": "Polish", "native": "Polski", "flag": "🇵🇱"},
    "el": {"name": "Greek", "native": "Ελληνικά", "flag": "🇬🇷"},
    "th": {"name": "Thai", "native": "ไทย", "flag": "🇹🇭"},
    "vi": {"name": "Vietnamese", "native": "Tiếng Việt", "flag": "🇻🇳"},
    "bg": {"name": "Bulgarian", "native": "Български", "flag": "🇧🇬"},
    "sw": {"name": "Swahili", "native": "Kiswahili", "flag": "🇰🇪"},
    "ur": {"name": "Urdu", "native": "اردو", "flag": "🇵🇰"},
}

# Global classifier
classifier = None


def load_model():
    """Load text-classification model fine-tuned for language detection"""
    global classifier
    
    print(f"📦 Loading language detection model: {MODEL_NAME}...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,  # Return all scores
        device=-1  # CPU
    )
    print("✅ Model loaded successfully!")
    return classifier


def detect_language(text, top_k=5):
    """
    Detect language using XLM-RoBERTa language detection model
    
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
    
    try:
        # Run classification - model returns list of dicts or nested list
        results = classifier(text)
        
        # Handle nested list structure (when top_k=None returns [[{...}, {...}]])
        if results and isinstance(results[0], list):
            results = results[0]
        
        # Sort by score descending and take top_k
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # Format results
        detections = []
        for item in results:
            lang_code = item['label']
            score = item['score']
            info = LANGUAGE_INFO.get(lang_code, {"name": lang_code.upper(), "native": lang_code, "flag": "🏳️"})
            
            detections.append({
                'language': info['name'],
                'confidence': round(score * 100, 2),
                'code': lang_code,
                'native_name': info['native'],
                'flag': info['flag']
            })
        
        return detections
        
    except Exception as e:
        print(f"Error detecting language: {e}")
        import traceback
        traceback.print_exc()
        return []


def split_by_script(text):
    """
    Split text into segments by script/language boundaries.
    Detects transitions between different writing systems.
    """
    if not text.strip():
        return []
    
    def get_char_type(char):
        """Determine the script type of a character"""
        code = ord(char)
        
        # Japanese (Hiragana, Katakana, Kanji) + Japanese punctuation
        if (0x3040 <= code <= 0x309F or  # Hiragana
            0x30A0 <= code <= 0x30FF or  # Katakana
            0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs (Kanji/Chinese)
            0x3400 <= code <= 0x4DBF or  # CJK Extension A
            0x3000 <= code <= 0x303F or  # CJK Punctuation (、。「」etc.)
            0xFF00 <= code <= 0xFFEF):   # Fullwidth forms (！？etc.)
            return "cjk"
        
        # Korean (Hangul)
        if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            return "korean"
        
        # Arabic
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            return "arabic"
        
        # Hebrew
        if 0x0590 <= code <= 0x05FF:
            return "hebrew"
        
        # Cyrillic (Russian, Ukrainian, etc.)
        if 0x0400 <= code <= 0x04FF:
            return "cyrillic"
        
        # Greek
        if 0x0370 <= code <= 0x03FF:
            return "greek"
        
        # Thai
        if 0x0E00 <= code <= 0x0E7F:
            return "thai"
        
        # Devanagari (Hindi)
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        
        # Latin (English, Spanish, French, German, etc.)
        if (0x0041 <= code <= 0x007A or  # Basic Latin letters
            0x00C0 <= code <= 0x00FF or  # Latin-1 Supplement
            0x0100 <= code <= 0x017F):   # Latin Extended-A
            return "latin"
        
        # Basic punctuation and whitespace - neutral (attach to previous)
        if char in ' \t\n\r':
            return None
        
        # ASCII punctuation - neutral
        if 0x0020 <= code <= 0x0040 or 0x005B <= code <= 0x0060 or 0x007B <= code <= 0x007F:
            return None
        
        return "other"
    
    segments = []
    current_segment = ""
    current_type = None
    pending_neutral = ""  # Buffer for neutral chars between scripts
    
    for char in text:
        char_type = get_char_type(char)
        
        # Neutral characters - buffer them
        if char_type is None:
            pending_neutral += char
            continue
        
        # Same script type - continue segment (include buffered neutrals)
        if current_type is None or current_type == char_type:
            current_segment += pending_neutral + char
            pending_neutral = ""
            current_type = char_type
        else:
            # Script change detected - save current segment and start new
            if current_segment.strip():
                segments.append(current_segment.strip())
            current_segment = char
            pending_neutral = ""
            current_type = char_type
    
    # Add final segment (include any trailing neutral chars)
    final = (current_segment + pending_neutral).strip()
    if final:
        segments.append(final)
    
    # Filter out very short segments (likely just punctuation)
    segments = [s for s in segments if len(s.strip()) >= 3]
    
    return segments if segments else [text.strip()]


def detect_batch(texts, top_k=3):
    """Detect languages for multiple texts with smart splitting"""
    results = []
    
    for text in texts:
        # Check if text contains multiple scripts
        segments = split_by_script(text)
        
        # If multiple segments detected, process each
        if len(segments) > 1:
            for segment in segments:
                detection = detect_language(segment, top_k)
                results.append({
                    'text': segment[:100] + '...' if len(segment) > 100 else segment,
                    'detections': detection
                })
        else:
            # Single language text
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
    # Pre-load model at startup for faster first request
    print("🚀 Pre-loading model at startup...")
    load_model()
    print("✅ Ready to detect languages!")
    app.run(debug=True, port=5000)
