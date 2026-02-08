"""
Day 53: Text-to-Speech Notes Maker
Generate notes with LLM, convert to speech with TTS
"""

from flask import Flask, render_template, request, jsonify, send_file
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import os
import uuid
import time
import re
from datetime import datetime

app = Flask(__name__)

# Ensure audio directory exists
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

class NotesGenerator:
    def __init__(self):
        print("🚀 Initializing Notes Generator...")
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load GPT-2 model for text generation"""
        try:
            print("📦 Loading GPT-2 model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Set pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            print("✅ GPT-2 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def generate_notes(self, topic: str, style: str = "detailed", length: str = "medium") -> dict:
        """Generate notes on a topic using GPT-2"""
        start_time = time.time()
        
        # Create prompt based on style
        prompts = {
            "detailed": f"Comprehensive notes on {topic}:\n\n1. Introduction:\n{topic} is",
            "bullet": f"Key points about {topic}:\n\n• {topic} refers to",
            "summary": f"Brief summary of {topic}:\n\n{topic} is",
            "study": f"Study notes for {topic}:\n\nDefinition: {topic} is"
        }
        
        prompt = prompts.get(style, prompts["detailed"])
        
        # Determine max length based on selection
        length_map = {
            "short": 150,
            "medium": 300,
            "long": 500
        }
        max_new_tokens = length_map.get(length, 300)
        
        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the generated text
            notes = self._clean_notes(generated_text, topic)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "success": True,
                "notes": notes,
                "topic": topic,
                "style": style,
                "length": length,
                "word_count": len(notes.split()),
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _clean_notes(self, text: str, topic: str) -> str:
        """Clean and format the generated notes"""
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1:
            # Keep complete sentences
            complete_sentences = sentences[:-1]  # Remove last potentially incomplete
            text = '. '.join(complete_sentences) + '.'
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()


class TextToSpeech:
    def __init__(self):
        print("🔊 Initializing Text-to-Speech engine...")
    
    def convert_to_speech(self, text: str, language: str = "en", speed: bool = False) -> dict:
        """Convert text to speech using gTTS"""
        start_time = time.time()
        
        try:
            # Generate unique filename
            filename = f"notes_{uuid.uuid4().hex[:8]}_{int(time.time())}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            # Create TTS
            tts = gTTS(text=text, lang=language, slow=speed)
            tts.save(filepath)
            
            processing_time = round(time.time() - start_time, 2)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            
            return {
                "success": True,
                "filename": filename,
                "filepath": f"/static/audio/{filename}",
                "file_size": file_size_str,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Initialize components
print("=" * 50)
print("🎙️ Text-to-Speech Notes Maker - Day 53")
print("=" * 50)

notes_generator = NotesGenerator()
tts_engine = TextToSpeech()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """Generate notes for a topic"""
    data = request.json
    topic = data.get('topic', '').strip()
    style = data.get('style', 'detailed')
    length = data.get('length', 'medium')
    
    if not topic:
        return jsonify({"error": "Please provide a topic"}), 400
    
    if len(topic) > 200:
        return jsonify({"error": "Topic is too long (max 200 characters)"}), 400
    
    result = notes_generator.generate_notes(topic, style, length)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result.get("error", "Failed to generate notes")}), 500


@app.route('/convert-to-speech', methods=['POST'])
def convert_to_speech():
    """Convert text to speech"""
    data = request.json
    text = data.get('text', '').strip()
    language = data.get('language', 'en')
    slow = data.get('slow', False)
    
    if not text:
        return jsonify({"error": "Please provide text to convert"}), 400
    
    if len(text) > 5000:
        return jsonify({"error": "Text is too long (max 5000 characters)"}), 400
    
    result = tts_engine.convert_to_speech(text, language, slow)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result.get("error", "Failed to convert to speech")}), 500


@app.route('/generate-and-speak', methods=['POST'])
def generate_and_speak():
    """Generate notes and convert to speech in one call"""
    data = request.json
    topic = data.get('topic', '').strip()
    style = data.get('style', 'detailed')
    length = data.get('length', 'medium')
    language = data.get('language', 'en')
    slow = data.get('slow', False)
    
    if not topic:
        return jsonify({"error": "Please provide a topic"}), 400
    
    # Generate notes
    notes_result = notes_generator.generate_notes(topic, style, length)
    
    if not notes_result["success"]:
        return jsonify({"error": notes_result.get("error", "Failed to generate notes")}), 500
    
    # Convert to speech
    tts_result = tts_engine.convert_to_speech(notes_result["notes"], language, slow)
    
    if not tts_result["success"]:
        return jsonify({"error": tts_result.get("error", "Failed to convert to speech")}), 500
    
    # Combine results
    return jsonify({
        "success": True,
        "topic": topic,
        "notes": notes_result["notes"],
        "style": style,
        "length": length,
        "word_count": notes_result["word_count"],
        "notes_time": notes_result["processing_time"],
        "audio_file": tts_result["filepath"],
        "audio_size": tts_result["file_size"],
        "tts_time": tts_result["processing_time"],
        "total_time": round(notes_result["processing_time"] + tts_result["processing_time"], 2)
    })


@app.route('/download/<filename>')
def download_audio(filename):
    """Download audio file"""
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    print("\n🌐 Starting server at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
