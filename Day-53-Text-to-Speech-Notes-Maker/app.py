from flask import Flask, render_template, request, jsonify, send_file, session
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
import os
import uuid
import time
import re
import json
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directories
BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, 'static', 'audio')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# CONFIGURATION & PRESETS

TONE_PRESETS = {
    "academic": {
        "prefix": "In academic terms, ",
        "style": "formal, precise, and scholarly",
        "temp": 0.6
    },
    "casual": {
        "prefix": "Simply put, ",
        "style": "friendly, easy to understand, conversational",
        "temp": 0.8
    },
    "motivational": {
        "prefix": "Here's an inspiring take: ",
        "style": "encouraging, uplifting, and energizing",
        "temp": 0.85
    },
    "exam_focused": {
        "prefix": "For exam preparation: ",
        "style": "concise, definition-focused, with key points",
        "temp": 0.5
    }
}

DIFFICULTY_LEVELS = {
    "beginner": {
        "complexity": "simple vocabulary, basic concepts, many examples",
        "max_tokens": 250
    },
    "intermediate": {
        "complexity": "moderate detail, some technical terms explained",
        "max_tokens": 350
    },
    "advanced": {
        "complexity": "technical language, in-depth analysis, expert level",
        "max_tokens": 500
    }
}

STYLE_TEMPLATES = {
    "detailed": "Comprehensive notes on {topic}:\n\n1. Introduction:\n{topic} is",
    "bullet": "Key points about {topic}:\n\n• {topic} refers to",
    "summary": "Brief summary of {topic}:\n\n{topic} is",
    "study": "Study notes for {topic}:\n\nDefinition: {topic} is",
    "revision": "Quick revision notes on {topic}:\n\nKey facts about {topic}:",
    "exam": "Exam notes for {topic}:\n\nDefinition: {topic}\nKey formulas/concepts:",
    "flashcard": "Flashcard content for {topic}:\n\nQ: What is {topic}?\nA: {topic} is"
}

# Voice settings for different TTS options
VOICE_SETTINGS = {
    "female_us": {"lang": "en", "tld": "com"},
    "female_uk": {"lang": "en", "tld": "co.uk"},
    "female_au": {"lang": "en", "tld": "com.au"},
    "male_in": {"lang": "en", "tld": "co.in"},
    "neutral": {"lang": "en", "tld": "com"}
}


class NotesGenerator:
    def __init__(self):
        print("🚀 Initializing Enhanced Notes Generator...")
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load GPT-2 model for text generation"""
        try:
            print("📦 Loading GPT-2 model...")
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
                self.model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
            except Exception:
                print("⚠️ Cached model not found, downloading...")
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            print("✅ GPT-2 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def generate_notes(self, topic: str, options: dict) -> dict:
        """Generate notes with all enhancement options"""
        start_time = time.time()
        
        # Extract options
        style = options.get('style', 'detailed')
        tone = options.get('tone', 'academic')
        difficulty = options.get('difficulty', 'intermediate')
        custom_prompt = options.get('custom_prompt', '')
        
        # Get presets
        tone_preset = TONE_PRESETS.get(tone, TONE_PRESETS['academic'])
        diff_preset = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS['intermediate'])
        
        # Build prompt
        if custom_prompt:
            prompt = custom_prompt.replace('{topic}', topic)
        else:
            base_template = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES['detailed'])
            prompt = f"{tone_preset['prefix']}{base_template.format(topic=topic)}"
        
        max_tokens = diff_preset['max_tokens']
        temperature = tone_preset['temp']
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            notes = self._clean_notes(generated_text)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(notes, topic)
            
            # Generate headings/sections
            sections = self._generate_sections(notes)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "success": True,
                "notes": notes,
                "topic": topic,
                "key_concepts": key_concepts,
                "sections": sections,
                "word_count": len(notes.split()),
                "processing_time": processing_time,
                "options_used": {
                    "style": style,
                    "tone": tone,
                    "difficulty": difficulty
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_questions(self, notes: str, topic: str, q_type: str = "mcq") -> dict:
        """Generate questions from notes"""
        try:
            if q_type == "mcq":
                prompt = f"Generate 3 multiple choice questions about {topic}:\n\nBased on: {notes[:500]}\n\nQ1:"
            else:
                prompt = f"Generate 3 short answer questions about {topic}:\n\nBased on: {notes[:500]}\n\nQ1:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            questions_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            questions = self._parse_questions(questions_text, q_type)
            
            return {"success": True, "questions": questions, "type": q_type}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_flashcards(self, notes: str, topic: str) -> dict:
        """Generate flashcards from notes"""
        try:
            prompt = f"Create 5 flashcards (Q&A format) about {topic}:\n\nContent: {notes[:400]}\n\nCard 1:\nQ:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            flashcards_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            flashcards = self._parse_flashcards(flashcards_text)
            
            return {"success": True, "flashcards": flashcards}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def explain_simpler(self, text: str) -> dict:
        """Explain a piece of text more simply"""
        try:
            prompt = f"Explain this simply for a beginner:\n\n{text[:300]}\n\nSimple explanation:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanation = explanation.split("Simple explanation:")[-1].strip()
            
            return {"success": True, "explanation": self._clean_notes(explanation)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def compare_topics(self, topic1: str, topic2: str) -> dict:
        """Compare two topics"""
        try:
            prompt = f"Compare and contrast {topic1} vs {topic2}:\n\nSimilarities:\n- Both"
            
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            comparison = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "comparison": self._clean_notes(comparison),
                "topic1": topic1,
                "topic2": topic2
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _clean_notes(self, text: str) -> str:
        """Clean and format generated notes"""
        sentences = text.split('.')
        if len(sentences) > 1:
            complete_sentences = sentences[:-1]
            text = '. '.join(complete_sentences) + '.'
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def _extract_key_concepts(self, notes: str, topic: str) -> list:
        """Extract key concepts/terms from notes"""
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', notes)
        concepts = list(set([w for w in words if len(w) > 3 and w.lower() != topic.lower()]))
        return concepts[:10]
    
    def _generate_sections(self, notes: str) -> list:
        """Generate section headings for notes"""
        sections = []
        paragraphs = notes.split('\n\n')
        for i, para in enumerate(paragraphs):
            if para.strip():
                words = para.split()[:5]
                heading = ' '.join(words) + '...' if len(words) >= 5 else ' '.join(words)
                sections.append({
                    "id": i + 1,
                    "heading": heading,
                    "content": para.strip(),
                    "word_count": len(para.split())
                })
        return sections
    
    def _parse_questions(self, text: str, q_type: str) -> list:
        """Parse generated questions"""
        questions = []
        q_pattern = re.findall(r'Q\d*[:\.]?\s*(.+?)(?=Q\d|$)', text, re.DOTALL)
        for i, q in enumerate(q_pattern[:5]):
            questions.append({
                "id": i + 1,
                "question": q.strip()[:200],
                "type": q_type
            })
        return questions if questions else [{"id": 1, "question": "What is the main concept discussed?", "type": q_type}]
    
    def _parse_flashcards(self, text: str) -> list:
        """Parse generated flashcards"""
        flashcards = []
        cards = re.split(r'Card \d+:', text)
        for i, card in enumerate(cards[1:6]):
            parts = card.split('A:')
            if len(parts) >= 2:
                q = parts[0].replace('Q:', '').strip()[:150]
                a = parts[1].strip()[:200]
                flashcards.append({"id": i + 1, "question": q, "answer": a})
        
        if not flashcards:
            flashcards = [{"id": 1, "question": "What is this topic about?", "answer": "Review your notes for the answer."}]
        return flashcards


class TextToSpeech:
    def __init__(self):
        print("🔊 Initializing Enhanced Text-to-Speech engine...")
    
    def convert_to_speech(self, text: str, options: dict) -> dict:
        """Convert text to speech with enhanced options"""
        start_time = time.time()
        
        language = options.get('language', 'en')
        slow = options.get('slow', False)
        voice = options.get('voice', 'female_us')
        
        voice_setting = VOICE_SETTINGS.get(voice, VOICE_SETTINGS['female_us'])
        
        try:
            filename = f"notes_{uuid.uuid4().hex[:8]}_{int(time.time())}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            tts = gTTS(
                text=text, 
                lang=language if language in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh-CN', 'hi'] else 'en',
                slow=slow,
                tld=voice_setting.get('tld', 'com')
            )
            tts.save(filepath)
            
            processing_time = round(time.time() - start_time, 2)
            file_size = os.path.getsize(filepath)
            file_size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            
            word_count = len(text.split())
            duration_mins = word_count / 150
            duration_str = f"{int(duration_mins)}:{int((duration_mins % 1) * 60):02d}"
            
            sentences = self._split_sentences(text)
            timestamps = self._generate_timestamps(sentences, duration_mins * 60)
            
            return {
                "success": True,
                "filename": filename,
                "filepath": f"/static/audio/{filename}",
                "file_size": file_size_str,
                "duration": duration_str,
                "processing_time": processing_time,
                "sentences": sentences,
                "timestamps": timestamps
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_pomodoro_sessions(self, text: str, session_count: int = 4) -> dict:
        """Split notes into Pomodoro study sessions"""
        sentences = self._split_sentences(text)
        total_sentences = len(sentences)
        per_session = max(1, total_sentences // session_count)
        
        sessions = []
        for i in range(session_count):
            start = i * per_session
            end = start + per_session if i < session_count - 1 else total_sentences
            session_text = ' '.join(sentences[start:end])
            if session_text.strip():
                sessions.append({
                    "id": i + 1,
                    "text": session_text,
                    "word_count": len(session_text.split())
                })
        
        return {
            "success": True,
            "sessions": sessions,
            "session_count": len(sessions)
        }
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_timestamps(self, sentences: list, total_duration: float) -> list:
        """Generate approximate timestamps for each sentence"""
        if not sentences:
            return []
        
        total_words = sum(len(s.split()) for s in sentences)
        timestamps = []
        current_time = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            duration = (words / total_words) * total_duration if total_words > 0 else 0
            timestamps.append({
                "start": round(current_time, 2),
                "end": round(current_time + duration, 2),
                "text": sentence
            })
            current_time += duration
        
        return timestamps


class StudyTracker:
    """Track study progress and manage bookmarks"""
    
    def __init__(self):
        self.data_file = os.path.join(DATA_DIR, 'study_progress.json')
        self.load_data()
    
    def load_data(self):
        """Load study data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.data = json.load(f)
            else:
                self.data = {"topics": {}, "bookmarks": [], "challenges": [], "streak": 0, "last_study": None}
        except:
            self.data = {"topics": {}, "bookmarks": [], "challenges": [], "streak": 0, "last_study": None}
    
    def save_data(self):
        """Save study data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def track_topic(self, topic: str):
        """Track a studied topic"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if topic not in self.data["topics"]:
            self.data["topics"][topic] = {"count": 0, "last_studied": None, "dates": []}
        
        self.data["topics"][topic]["count"] += 1
        self.data["topics"][topic]["last_studied"] = today
        if today not in self.data["topics"][topic]["dates"]:
            self.data["topics"][topic]["dates"].append(today)
        
        # Update streak
        if self.data["last_study"] != today:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            if self.data["last_study"] == yesterday:
                self.data["streak"] += 1
            elif self.data["last_study"] != today:
                self.data["streak"] = 1
            self.data["last_study"] = today
        
        self.save_data()
        return self.get_progress()
    
    def add_bookmark(self, topic: str, timestamp: float, note: str = ""):
        """Add a bookmark"""
        bookmark = {
            "id": str(uuid.uuid4())[:8],
            "topic": topic,
            "timestamp": timestamp,
            "note": note,
            "created": datetime.now().isoformat()
        }
        self.data["bookmarks"].append(bookmark)
        self.save_data()
        return bookmark
    
    def get_bookmarks(self, topic: str = None):
        """Get bookmarks"""
        if topic:
            return [b for b in self.data["bookmarks"] if b["topic"] == topic]
        return self.data["bookmarks"]
    
    def remove_bookmark(self, bookmark_id: str):
        """Remove a bookmark"""
        self.data["bookmarks"] = [b for b in self.data["bookmarks"] if b["id"] != bookmark_id]
        self.save_data()
    
    def get_daily_challenge(self):
        """Get daily topic challenge"""
        challenges = [
            "Machine Learning", "Quantum Physics", "Climate Change", "Ancient Rome",
            "Blockchain", "Human Psychology", "Solar System", "Artificial Intelligence",
            "World War II", "Genetic Engineering", "Philosophy", "Cryptocurrency",
            "Renaissance Art", "Cybersecurity", "Evolution", "Black Holes",
            "Neural Networks", "Photosynthesis", "Democracy", "Relativity Theory"
        ]
        
        today = datetime.now().strftime("%Y-%m-%d")
        index = int(hashlib.md5(today.encode()).hexdigest(), 16) % len(challenges)
        
        completed_dates = [c.get("date") for c in self.data.get("challenges", [])]
        
        return {
            "topic": challenges[index],
            "date": today,
            "completed": today in completed_dates
        }
    
    def complete_challenge(self, topic: str):
        """Mark daily challenge as complete"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.data["challenges"].append({"topic": topic, "date": today})
        self.save_data()
    
    def get_progress(self):
        """Get overall study progress"""
        return {
            "total_topics": len(self.data["topics"]),
            "total_sessions": sum(t["count"] for t in self.data["topics"].values()),
            "streak": self.data["streak"],
            "bookmarks_count": len(self.data["bookmarks"]),
            "recent_topics": list(self.data["topics"].keys())[-5:]
        }


# Initialize components
print("=" * 50)
print("🎙️ Text-to-Speech Notes Maker - Enhanced Edition")
print("=" * 50)

notes_generator = NotesGenerator()
tts_engine = TextToSpeech()
study_tracker = StudyTracker()


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """Generate notes with all options"""
    data = request.json
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({"error": "Please provide a topic"}), 400
    
    options = {
        'style': data.get('style', 'detailed'),
        'tone': data.get('tone', 'academic'),
        'difficulty': data.get('difficulty', 'intermediate'),
        'custom_prompt': data.get('custom_prompt', '')
    }
    
    result = notes_generator.generate_notes(topic, options)
    
    if result["success"]:
        study_tracker.track_topic(topic)
        return jsonify(result)
    return jsonify({"error": result.get("error", "Failed to generate notes")}), 500


@app.route('/convert-to-speech', methods=['POST'])
def convert_to_speech():
    """Convert text to speech with options"""
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "Please provide text"}), 400
    
    options = {
        'language': data.get('language', 'en'),
        'slow': data.get('slow', False),
        'voice': data.get('voice', 'female_us')
    }
    
    result = tts_engine.convert_to_speech(text, options)
    
    if result["success"]:
        return jsonify(result)
    return jsonify({"error": result.get("error", "Failed to convert")}), 500


@app.route('/generate-and-speak', methods=['POST'])
def generate_and_speak():
    """Generate notes and convert to speech"""
    data = request.json
    topic = data.get('topic', '').strip()
    mode = data.get('mode', 'both')  # 'notes' or 'both'
    
    if not topic:
        return jsonify({"error": "Please provide a topic"}), 400
    
    notes_options = {
        'style': data.get('style', 'detailed'),
        'tone': data.get('tone', 'academic'),
        'difficulty': data.get('difficulty', 'intermediate'),
        'custom_prompt': data.get('custom_prompt', '')
    }
    
    notes_result = notes_generator.generate_notes(topic, notes_options)
    
    if not notes_result["success"]:
        return jsonify({"error": notes_result.get("error")}), 500
    
    # Only generate audio if mode is 'both'
    tts_result = None
    if mode == 'both':
        tts_options = {
            'language': data.get('language', 'en'),
            'slow': data.get('slow', False),
            'voice': data.get('voice', 'female_us')
        }
        
        tts_result = tts_engine.convert_to_speech(notes_result["notes"], tts_options)
        
        if not tts_result["success"]:
            return jsonify({"error": tts_result.get("error")}), 500
    
    study_tracker.track_topic(topic)
    
    response = {
        "success": True,
        "topic": topic,
        "notes": notes_result["notes"],
        "key_concepts": notes_result["key_concepts"],
        "sections": notes_result["sections"],
        "word_count": notes_result["word_count"],
        "notes_time": notes_result["processing_time"],
        "options": notes_options
    }
    
    if tts_result:
        response.update({
            "audio_file": tts_result["filepath"],
            "audio_size": tts_result["file_size"],
            "duration": tts_result["duration"],
            "tts_time": tts_result["processing_time"],
            "sentences": tts_result["sentences"],
            "timestamps": tts_result["timestamps"],
            "total_time": round(notes_result["processing_time"] + tts_result["processing_time"], 2),
        })
    else:
        response.update({
            "audio_file": "",
            "audio_size": "-",
            "duration": "-",
            "total_time": round(notes_result["processing_time"], 2),
        })
    
    return jsonify(response)


@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Generate questions from notes"""
    data = request.json
    notes = data.get('notes', '')
    topic = data.get('topic', '')
    q_type = data.get('type', 'mcq')
    
    if not notes:
        return jsonify({"error": "Please provide notes"}), 400
    
    result = notes_generator.generate_questions(notes, topic, q_type)
    return jsonify(result)


@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards from notes"""
    data = request.json
    notes = data.get('notes', '')
    topic = data.get('topic', '')
    
    if not notes:
        return jsonify({"error": "Please provide notes"}), 400
    
    result = notes_generator.generate_flashcards(notes, topic)
    return jsonify(result)


@app.route('/explain-simpler', methods=['POST'])
def explain_simpler():
    """Explain text more simply"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Please provide text"}), 400
    
    result = notes_generator.explain_simpler(text)
    return jsonify(result)


@app.route('/compare-topics', methods=['POST'])
def compare_topics():
    """Compare two topics"""
    data = request.json
    topic1 = data.get('topic1', '').strip()
    topic2 = data.get('topic2', '').strip()
    
    if not topic1 or not topic2:
        return jsonify({"error": "Please provide both topics"}), 400
    
    result = notes_generator.compare_topics(topic1, topic2)
    return jsonify(result)


@app.route('/pomodoro-split', methods=['POST'])
def pomodoro_split():
    """Split notes into Pomodoro sessions"""
    data = request.json
    text = data.get('text', '')
    session_count = data.get('session_count', 4)
    
    if not text:
        return jsonify({"error": "Please provide text"}), 400
    
    result = tts_engine.create_pomodoro_sessions(text, session_count)
    return jsonify(result)


@app.route('/bookmark', methods=['POST'])
def add_bookmark():
    """Add a bookmark"""
    data = request.json
    topic = data.get('topic', '')
    timestamp = data.get('timestamp', 0)
    note = data.get('note', '')
    
    bookmark = study_tracker.add_bookmark(topic, timestamp, note)
    return jsonify({"success": True, "bookmark": bookmark})


@app.route('/bookmarks', methods=['GET'])
def get_bookmarks():
    """Get all bookmarks"""
    topic = request.args.get('topic')
    bookmarks = study_tracker.get_bookmarks(topic)
    return jsonify({"bookmarks": bookmarks})


@app.route('/bookmark/<bookmark_id>', methods=['DELETE'])
def remove_bookmark(bookmark_id):
    """Remove a bookmark"""
    study_tracker.remove_bookmark(bookmark_id)
    return jsonify({"success": True})


@app.route('/progress', methods=['GET'])
def get_progress():
    """Get study progress"""
    progress = study_tracker.get_progress()
    return jsonify(progress)


@app.route('/daily-challenge', methods=['GET'])
def get_daily_challenge():
    """Get daily topic challenge"""
    challenge = study_tracker.get_daily_challenge()
    return jsonify(challenge)


@app.route('/complete-challenge', methods=['POST'])
def complete_challenge():
    """Complete daily challenge"""
    data = request.json
    topic = data.get('topic', '')
    study_tracker.complete_challenge(topic)
    return jsonify({"success": True})


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