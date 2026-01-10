from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import re
import os

app = Flask(__name__)
app.secret_key = 'day50-auto-corrector-secret-key'

# Grammar Correction Engine

class AutoCorrector:
    """
    LLM-powered text correction engine using T5.
    
    Features:
    - Grammar correction
    - Spelling correction
    - Punctuation fixes
    - Capitalization fixes
    - Context-aware corrections
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.corrector = None
        self.is_ready = False
        self.model_name = "vennify/t5-base-grammar-correction"
        self.stats = {
            'total_corrections': 0,
            'total_characters': 0,
            'sessions': 0
        }
    
    def initialize(self):
        """Initialize the grammar correction model."""
        try:
            print(f"🔧 Loading model: {self.model_name}")
            print("⏳ This may take a moment on first run...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.corrector = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512
            )
            
            # Test the model
            test_result = self.correct("This is a tset.")
            if test_result:
                print(f"✅ Test passed: 'This is a tset.' → '{test_result}'")
            
            self.is_ready = True
            print("✅ Auto-Corrector initialized successfully!")
            return True, "Model loaded successfully!"
            
        except Exception as e:
            print(f"❌ Error initializing model: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def correct(self, text):
        """
        Correct grammar and spelling in the given text.
        
        Args:
            text: Input text with potential errors
            
        Returns:
            Corrected text
        """
        if not self.is_ready:
            return text
        
        try:
            # Handle empty input
            if not text or not text.strip():
                return text
            
            # Split into sentences for better correction
            sentences = self._split_sentences(text)
            corrected_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    # Add prefix for T5 grammar correction
                    input_text = f"grammar: {sentence}"
                    
                    # Generate correction
                    result = self.corrector(
                        input_text,
                        max_length=len(sentence) + 50,
                        num_return_sequences=1,
                        do_sample=False
                    )
                    
                    corrected = result[0]['generated_text']
                    corrected_sentences.append(corrected)
            
            corrected_text = ' '.join(corrected_sentences)
            
            # Update stats
            self.stats['total_corrections'] += 1
            self.stats['total_characters'] += len(text)
            
            return corrected_text
            
        except Exception as e:
            print(f"Error during correction: {str(e)}")
            return text
    
    def _split_sentences(self, text):
        """Split text into sentences for processing."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_diff(self, original, corrected):
        """
        Get the differences between original and corrected text.
        
        Returns list of changes made.
        """
        changes = []
        
        # Simple word-level diff
        original_words = original.split()
        corrected_words = corrected.split()
        
        # Find differences
        max_len = max(len(original_words), len(corrected_words))
        
        for i in range(max_len):
            orig = original_words[i] if i < len(original_words) else ""
            corr = corrected_words[i] if i < len(corrected_words) else ""
            
            if orig.lower() != corr.lower():
                changes.append({
                    'original': orig,
                    'corrected': corr,
                    'position': i
                })
        
        return changes
    
    def get_stats(self):
        """Get correction statistics."""
        return {
            'model': self.model_name,
            'is_ready': self.is_ready,
            'total_corrections': self.stats['total_corrections'],
            'total_characters': self.stats['total_characters']
        }


# Initialize the corrector
corrector = AutoCorrector()

# ============================================
# Flask Routes
# ============================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/correct', methods=['POST'])
def correct_text():
    """API endpoint to correct text."""
    if not corrector.is_ready:
        return jsonify({
            'success': False,
            'error': 'Model not loaded yet. Please wait...'
        })
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({
            'success': False,
            'error': 'Please enter some text to correct.'
        })
    
    try:
        # Perform correction
        corrected = corrector.correct(text)
        
        # Get changes
        changes = corrector.get_diff(text, corrected)
        
        return jsonify({
            'success': True,
            'original': text,
            'corrected': corrected,
            'changes': changes,
            'change_count': len(changes)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/status')
def status():
    """Check if the model is ready."""
    return jsonify({
        'ready': corrector.is_ready,
        'stats': corrector.get_stats()
    })


@app.route('/examples')
def examples():
    """Get example texts with errors."""
    examples = [
        {
            'id': 1,
            'text': "I cant beleive how grate this is working.",
            'description': "Spelling & contractions"
        },
        {
            'id': 2,
            'text': "their going to the store but there car is broken",
            'description': "Homophones (their/they're/there)"
        },
        {
            'id': 3,
            'text': "me and him went to the park yesterday it was very fun",
            'description': "Grammar & punctuation"
        },
        {
            'id': 4,
            'text': "The quick brown fox jump over the lazy dogs.",
            'description': "Subject-verb agreement"
        },
        {
            'id': 5,
            'text': "i dont know weather i should go too the party or not",
            'description': "Multiple errors"
        }
    ]
    return jsonify(examples)


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Day 50: Auto-Corrector (LLM-Powered)               ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Model: T5 Grammar Correction                                ║
    ║  Features: Spelling, Grammar, Punctuation                    ║
    ║  Framework: Hugging Face Transformers                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize the model
    print("🚀 Starting Auto-Corrector...")
    success, message = corrector.initialize()
    
    if success:
        print(f"\n🌐 Starting server at http://localhost:5000\n")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    else:
        print(f"❌ Failed to initialize: {message}")
