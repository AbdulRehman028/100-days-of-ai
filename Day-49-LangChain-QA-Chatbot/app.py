"""
Day 49: LangChain Q&A Chatbot
=============================
A Q&A chatbot built with LangChain using ChatModel architecture.
Uses Hugging Face's inference API for powerful language understanding.

LangChain Components:
- ChatHuggingFace: Chat model wrapper for Hugging Face
- ChatPromptTemplate: Structured prompts for chat
- LLMChain / LCEL: Chain for connecting components
"""

from flask import Flask, render_template, request, jsonify, session
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import uuid
import os

app = Flask(__name__)
app.secret_key = 'day49-langchain-qa-chatbot-secret-key'

# ============================================
# LangChain Q&A Chatbot Engine
# ============================================

class LangChainQAChatbot:
    """
    Q&A Chatbot using LangChain with ChatModel.
    
    Architecture:
    User Input → ChatPromptTemplate → ChatModel → StrOutputParser → Response
    
    Uses LCEL (LangChain Expression Language) for clean chain composition.
    """
    
    def __init__(self):
        self.llm = None
        self.chat_model = None
        self.chain = None
        self.conversations = {}
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.is_ready = False
        
    def initialize(self, hf_token=None):
        """
        Initialize LangChain components with Hugging Face ChatModel.
        
        Chain Architecture:
        1. ChatPromptTemplate - Formats messages with system prompt
        2. ChatHuggingFace - The actual chat model
        3. StrOutputParser - Extracts string response
        """
        try:
            # Get token from parameter or environment
            token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
            if not token:
                return False, "Please provide a Hugging Face API token"
            
            print(f"🔗 Initializing LangChain with model: {self.model_name}")
            
            # Step 1: Create the base LLM endpoint
            self.llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                huggingfacehub_api_token=token,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
            )
            
            # Step 2: Wrap as ChatModel for proper chat interface
            self.chat_model = ChatHuggingFace(
                llm=self.llm,
                verbose=True
            )
            
            # Step 3: Create the prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful, friendly AI assistant. You provide clear, accurate, and helpful answers to questions.

Guidelines:
- Be concise but thorough
- If you don't know something, say so honestly
- Use examples when helpful
- Be conversational and friendly"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Step 4: Build the chain using LCEL (LangChain Expression Language)
            self.chain = self.prompt | self.chat_model | StrOutputParser()
            
            self.is_ready = True
            print("✅ LangChain Q&A Chatbot initialized successfully!")
            return True, "Chatbot initialized successfully!"
            
        except Exception as e:
            print(f"❌ Error initializing chatbot: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_or_create_session(self, session_id):
        """Get or create a conversation session with message history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history': [],  # List of HumanMessage/AIMessage objects
                'messages': [],  # Plain text history for UI
                'created_at': datetime.now().isoformat()
            }
        return self.conversations[session_id]
    
    def ask(self, question, session_id):
        """
        Ask a question and get a response using the LangChain chain.
        
        Flow:
        1. Get conversation history
        2. Invoke chain with input + history
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
            
            # Invoke the chain with history
            response = self.chain.invoke({
                "input": question,
                "history": conv['history']
            })
            
            # Update conversation history with LangChain message objects
            conv['history'].append(HumanMessage(content=question))
            conv['history'].append(AIMessage(content=response))
            
            # Keep history manageable (last 10 exchanges)
            if len(conv['history']) > 20:
                conv['history'] = conv['history'][-20:]
            
            # Store plain text for UI
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
                'history': [],
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
            'model_type': 'ChatModel (Instruction-tuned)',
            'active_sessions': len(self.conversations),
            'total_messages': sum(
                len(conv['messages']) 
                for conv in self.conversations.values()
            ),
            'is_ready': self.is_ready
        }

# Initialize chatbot instance
chatbot = LangChainQAChatbot()

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
    ║  Framework: LangChain with ChatModel                         ║
    ║  Model: Mistral-7B-Instruct (via Hugging Face)               ║
    ║  Architecture: ChatPromptTemplate → ChatModel → Parser       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for pre-configured token
    if os.getenv("HF_TOKEN"):
        print("🔑 Found HF_TOKEN in environment, auto-initializing...")
        chatbot.initialize()
    else:
        print("💡 Set your Hugging Face token in the web interface to start!")
    
    print("\n🌐 Starting server at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
