"""
Day 48: Chatbot with LLM/RNN
============================
A conversational chatbot using DialoGPT (GPT-2 based) from Hugging Face.
DialoGPT is trained on 147M Reddit conversations for open-domain dialogue.

This implements a seq2seq style conversation where:
- Input: User message + conversation history
- Output: Generated response maintaining context
"""

from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'day48-chatbot-llm-rnn-secret-key'

# Chatbot Engine using DialoGPT

class ConversationalChatbot:
    """
    Conversational chatbot using Microsoft's DialoGPT model.
    DialoGPT is a GPT-2 based model fine-tuned for dialogue generation.
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with DialoGPT model.
        
        Models available:
        - microsoft/DialoGPT-small (117M params) - Fast, less coherent
        - microsoft/DialoGPT-medium (345M params) - Good balance
        - microsoft/DialoGPT-large (762M params) - Best quality, slower
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversations = {}  # Store conversation history per session
        self.max_history_tokens = 1000  # Max tokens to keep in history
        
    def load_model(self):
        """Load the DialoGPT model and tokenizer."""
        print(f"📦 Loading model: {self.model_name}")
        print(f"🖥️  Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        return True
    
    def get_or_create_session(self, session_id):
        """Get or create a conversation session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history_ids': None,
                'messages': [],
                'created_at': datetime.now().isoformat()
            }
        return self.conversations[session_id]
    
    def check_intent(self, user_input):
        """
        Check for common intents that need predefined responses.
        DialoGPT doesn't know about itself, so we handle these specially.
        """
        text = user_input.lower().strip()
        
        # Greeting patterns
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(text.startswith(g) or text == g for g in greetings):
            return "greeting"
        
        # Capability questions
        capability_keywords = ['what can you do', 'what do you do', 'your capabilities', 
                              'help me with', 'what are you', 'who are you', 
                              'your purpose', 'your function', 'how can you help']
        if any(kw in text for kw in capability_keywords):
            return "capabilities"
        
        # Name questions
        name_keywords = ['your name', 'who are you', 'what are you called']
        if any(kw in text for kw in name_keywords):
            return "name"
        
        # How are you
        if 'how are you' in text or "how're you" in text:
            return "how_are_you"
        
        return None
    
    def get_intent_response(self, intent):
        """Get predefined response for detected intent."""
        responses = {
            'greeting': "Hello! 👋 I'm an AI chatbot powered by DialoGPT. I can have conversations on various topics, answer questions, discuss ideas, or just chat! Feel free to ask me anything.",
            
            'capabilities': "I'm a conversational AI that can:\n\n• 💬 Have natural conversations on any topic\n• 🤔 Answer questions and share knowledge\n• 💡 Discuss ideas and brainstorm\n• 📖 Tell stories and be creative\n• 🎭 Engage in roleplay scenarios\n\nI'm powered by DialoGPT, trained on millions of conversations. What would you like to talk about?",
            
            'name': "I'm DialoGPT Chatbot! 🤖 I'm powered by Microsoft's DialoGPT model, which is trained on 147 million Reddit conversations. I'm here to chat with you about anything!",
            
            'how_are_you': "I'm doing great, thank you for asking! 😊 As an AI, I'm always ready and eager to chat. How can I help you today?"
        }
        return responses.get(intent, None)
    
    def generate_response(self, user_input, session_id):
        """
        Generate a response using DialoGPT.
        
        The model uses the conversation history to maintain context.
        Each turn is encoded and concatenated to build context.
        """
        if self.model is None:
            return "Model not loaded. Please wait...", 0
        
        conv = self.get_or_create_session(session_id)
        
        # Check for common intents first
        intent = self.check_intent(user_input)
        if intent:
            response = self.get_intent_response(intent)
            # Store in history
            conv['messages'].append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            conv['messages'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            return response, 0.95
        
        # Encode the user input
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        ).to(self.device)
        
        # Append to conversation history
        if conv['history_ids'] is not None:
            # Concatenate history with new input
            bot_input_ids = torch.cat([conv['history_ids'], new_input_ids], dim=-1)
            
            # Trim if too long (keep recent context)
            if bot_input_ids.shape[-1] > self.max_history_tokens:
                bot_input_ids = bot_input_ids[:, -self.max_history_tokens:]
        else:
            bot_input_ids = new_input_ids
        
        # Create attention mask
        attention_mask = torch.ones_like(bot_input_ids)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                bot_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                min_length=5,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
        
        # Extract only the new tokens (the response)
        response_ids = output_ids[:, bot_input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Update conversation history
        conv['history_ids'] = output_ids
        
        # Store message in history
        conv['messages'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        conv['messages'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Calculate rough "confidence" based on response length and coherence
        confidence = min(1.0, len(response.split()) / 20)
        
        return response.strip(), confidence
    
    def clear_history(self, session_id):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            self.conversations[session_id] = {
                'history_ids': None,
                'messages': [],
                'created_at': datetime.now().isoformat()
            }
        return True
    
    def get_history(self, session_id):
        """Get conversation history for a session."""
        if session_id in self.conversations:
            return self.conversations[session_id]['messages']
        return []
    
    def get_stats(self):
        """Get chatbot statistics."""
        total_conversations = len(self.conversations)
        total_messages = sum(
            len(conv['messages']) 
            for conv in self.conversations.values()
        )
        return {
            'model': self.model_name,
            'device': self.device,
            'active_sessions': total_conversations,
            'total_messages': total_messages,
            'model_type': 'DialoGPT (GPT-2 based)',
            'architecture': 'Transformer Decoder (Autoregressive)'
        }

# Initialize chatbot
chatbot = ConversationalChatbot(model_name="microsoft/DialoGPT-medium")

# Flask Routes

@app.route('/')
def index():
    """Render the chat interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            })
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Generate response
        response, confidence = chatbot.generate_response(user_message, session_id)
        
        return jsonify({
            'success': True,
            'response': response,
            'confidence': confidence,
            'model': 'DialoGPT-medium'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation history."""
    session_id = session.get('session_id')
    if session_id:
        chatbot.clear_history(session_id)
    return jsonify({'success': True})

@app.route('/history', methods=['GET'])
def history():
    """Get conversation history."""
    session_id = session.get('session_id')
    if session_id:
        messages = chatbot.get_history(session_id)
        return jsonify({
            'success': True,
            'messages': messages
        })
    return jsonify({'success': True, 'messages': []})

@app.route('/stats', methods=['GET'])
def stats():
    """Get chatbot statistics."""
    return jsonify({
        'success': True,
        'stats': chatbot.get_stats()
    })

# Main Entry Point

if __name__ == '__main__':
    print("=" * 50)
    print("🤖 Day 48: Chatbot with LLM/RNN")
    print("=" * 50)
    print()
    
    # Load the model
    print("🚀 Initializing Chatbot Engine...")
    chatbot.load_model()
    
    print()
    print("=" * 50)
    print(f"📊 Model: {chatbot.model_name}")
    print(f"🖥️  Device: {chatbot.device}")
    print("=" * 50)
    print()
    
    # Start server
    print("🌐 Starting server at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
