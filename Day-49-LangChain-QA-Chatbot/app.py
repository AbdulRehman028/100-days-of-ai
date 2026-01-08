"""
Day 49: LangChain Q&A Chatbot
=============================
A Q&A chatbot built with LangChain using HuggingFace InferenceClient.
Uses Hugging Face's Inference API for powerful language understanding.

LangChain Components:
- Custom LLM wrapper with HF InferenceClient
- PromptTemplate: Structured prompts for chat
- LCEL: Chain for connecting components
"""

from flask import Flask, render_template, request, jsonify, session
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, List, Optional
import uuid
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = 'day49-langchain-qa-chatbot-secret-key'

# ============================================
# Custom LangChain LLM with HuggingFace
# ============================================

class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for HuggingFace InferenceClient."""
    
    client: Any = None
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    
    def __init__(self, token: str, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        super().__init__()
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=token)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the HuggingFace Inference API using chat completion."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content

# ============================================
# LangChain Q&A Chatbot Engine
# ============================================

class LangChainQAChatbot:
    """
    Q&A Chatbot using LangChain with HuggingFaceHub.
    
    Architecture:
    User Input → PromptTemplate → HuggingFaceHub LLM → StrOutputParser → Response
    
    Uses LCEL (LangChain Expression Language) for clean chain composition.
    """
    
    def __init__(self):
        self.llm = None
        self.chain = None
        self.conversations = {}
        self.model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.is_ready = False
        
    def initialize(self, hf_token=None):
        """
        Initialize LangChain components with HuggingFace Hub.
        
        Chain Architecture:
        1. PromptTemplate - Formats the question with context
        2. HuggingFaceHub - The LLM via HF Inference API
        3. StrOutputParser - Extracts string response
        """
        try:
            # Get token from parameter or environment
            token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
            if not token:
                return False, "Please provide a Hugging Face API token"
            
            # Set environment variable for HuggingFaceHub
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
            
            print(f"🔗 Initializing LangChain with model: {self.model_name}")
            
            # Step 1: Create custom LLM with HuggingFace InferenceClient
            self.llm = HuggingFaceLLM(token=token, model_name=self.model_name)
            
            # Step 2: Create the prompt template
            self.prompt = PromptTemplate(
                template="""You are a helpful AI assistant. Answer the following question clearly and helpfully.

Conversation History:
{history}

Question: {question}

Answer:""",
                input_variables=["history", "question"]
            )
            
            # Step 3: Build the chain using LCEL (LangChain Expression Language)
            # prompt | llm | parser
            self.chain = self.prompt | self.llm | StrOutputParser()
            
            # Test the chain
            print("🧪 Testing chain...")
            test_response = self.chain.invoke({
                "history": "",
                "question": "Say hello"
            })
            
            if test_response:
                print(f"✅ Test passed: {test_response[:50]}...")
            
            self.is_ready = True
            print("✅ LangChain Q&A Chatbot initialized successfully!")
            return True, "Chatbot initialized successfully!"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error initializing chatbot: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_or_create_session(self, session_id):
        """Get or create a conversation session with message history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],  # Plain text history for UI
                'created_at': datetime.now().isoformat()
            }
        return self.conversations[session_id]
    
    def format_history(self, messages):
        """Format conversation history as a string."""
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages[-6:]:  # Last 3 exchanges
            formatted.append(f"{msg['role'].title()}: {msg['content']}")
        return "\n".join(formatted)
    
    def ask(self, question, session_id):
        """
        Ask a question and get a response using the LangChain chain.
        
        Flow:
        1. Get conversation history
        2. Invoke chain with question + history
        3. Update history with new messages
        4. Return response
        """
        if not self.is_ready:
            return {
                'success': False,
                'error': 'Chatbot not initialized. Please set your HF token first.',
                'response': None
            }
        
        try:
            conv = self.get_or_create_session(session_id)
            
            # Format history for the prompt
            history_str = self.format_history(conv['messages'])
            
            # Invoke the chain
            response = self.chain.invoke({
                "history": history_str,
                "question": question
            })
            
            # Clean up response
            response = response.strip()
            
            # Store in history
            conv['messages'].append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            conv['messages'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep history manageable (last 10 exchanges)
            if len(conv['messages']) > 20:
                conv['messages'] = conv['messages'][-20:]
            
            return {
                'success': True,
                'response': response,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def clear_history(self, session_id):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            self.conversations[session_id] = {
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
        return {
            'model': self.model_name,
            'framework': 'LangChain',
            'model_type': 'Text2Text (Flan-T5)',
            'active_sessions': len(self.conversations),
            'total_messages': sum(
                len(conv['messages']) 
                for conv in self.conversations.values()
            ),
            'is_ready': self.is_ready
        }

# Initialize chatbot instance
chatbot = LangChainQAChatbot()

# Auto-initialize if token is in environment
if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("🔑 Found token in .env, auto-initializing...")
    chatbot.initialize()

# ============================================
# Flask Routes
# ============================================

@app.route('/')
def index():
    """Render the chat interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the chatbot with HF token."""
    data = request.get_json()
    token = data.get('token', '')
    
    if not token:
        return jsonify({
            'success': False,
            'message': 'Please provide a Hugging Face API token'
        })
    
    success, message = chatbot.initialize(hf_token=token)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Handle Q&A requests."""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({
            'success': False,
            'error': 'Please enter a question'
        })
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    result = chatbot.ask(question, session_id)
    
    return jsonify(result)

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
        return jsonify({
            'success': True,
            'history': chatbot.get_history(session_id)
        })
    return jsonify({'success': True, 'history': []})

@app.route('/stats', methods=['GET'])
def stats():
    """Get chatbot statistics."""
    return jsonify(chatbot.get_stats())

@app.route('/status', methods=['GET'])
def status():
    """Check if chatbot is ready."""
    return jsonify({
        'ready': chatbot.is_ready,
        'model': chatbot.model_name
    })

# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Day 49: LangChain Q&A Chatbot                      ║
    ║══════════════════════════════════════════════════════════════║
    ║  Framework: LangChain with LCEL                              ║
    ║  Model: Zephyr-7B-Beta (via Hugging Face)                    ║
    ║  Architecture: PromptTemplate → LLM → OutputParser           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for pre-configured token
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("🔑 Found HF token in environment, auto-initializing...")
        chatbot.initialize()
    else:
        print("💡 Set your Hugging Face token in the web interface to start!")
    
    print("\n🌐 Starting server at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
