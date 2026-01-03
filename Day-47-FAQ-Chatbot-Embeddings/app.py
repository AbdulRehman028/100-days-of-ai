"""
Day 47: FAQ Chatbot with Embeddings
Use embeddings to match queries to FAQ answers semantically.
Tech Stack: Python, Flask, Hugging Face Sentence Transformers
"""

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================

# Sentence Transformer model for embeddings
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and accurate model

# Similarity threshold for matching
SIMILARITY_THRESHOLD = 0.5

# ===============================
# FAQ KNOWLEDGE BASE
# ===============================

FAQ_DATA = [
    # General Questions
    {
        "category": "General",
        "question": "What is this chatbot?",
        "answer": "I'm an FAQ chatbot that uses AI embeddings to understand your questions and find the most relevant answers from our knowledge base. Unlike rule-based bots, I can understand the meaning of your questions even if you phrase them differently!"
    },
    {
        "category": "General",
        "question": "How does this chatbot work?",
        "answer": "I use Sentence Transformers to convert your question into a numerical vector (embedding). Then I compare it with embeddings of all FAQ questions using cosine similarity to find the best match. This allows me to understand semantic meaning, not just keywords!"
    },
    {
        "category": "General",
        "question": "What technology powers this chatbot?",
        "answer": "This chatbot is built with Python, Flask, and Hugging Face Sentence Transformers. It uses the 'all-MiniLM-L6-v2' model to create 384-dimensional embeddings for semantic similarity matching."
    },
    
    # About AI
    {
        "category": "AI & Technology",
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes learning from data, reasoning, problem-solving, perception, and language understanding. AI powers technologies like voice assistants, recommendation systems, and autonomous vehicles."
    },
    {
        "category": "AI & Technology",
        "question": "What are embeddings in machine learning?",
        "answer": "Embeddings are numerical representations of data (like text, images, or audio) in a continuous vector space. Similar items have similar embeddings. For text, embeddings capture semantic meaning, so 'happy' and 'joyful' would have similar vectors even though they're different words."
    },
    {
        "category": "AI & Technology",
        "question": "What is natural language processing?",
        "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It powers chatbots, translation services, sentiment analysis, and voice assistants like Siri and Alexa."
    },
    {
        "category": "AI & Technology",
        "question": "What is machine learning?",
        "answer": "Machine Learning (ML) is a subset of AI where computers learn patterns from data without being explicitly programmed. It includes supervised learning (with labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning from rewards)."
    },
    {
        "category": "AI & Technology",
        "question": "What is deep learning?",
        "answer": "Deep Learning is a subset of machine learning that uses neural networks with many layers. It excels at tasks like image recognition, speech processing, and language understanding. Popular frameworks include TensorFlow, PyTorch, and Keras."
    },
    
    # Programming
    {
        "category": "Programming",
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in AI, data science, web development, automation, and scientific computing. Its rich ecosystem includes libraries like NumPy, Pandas, TensorFlow, and Flask."
    },
    {
        "category": "Programming",
        "question": "What is Flask?",
        "answer": "Flask is a lightweight Python web framework. It's simple, flexible, and perfect for building APIs and web applications. Unlike Django, Flask gives you more control and is ideal for small to medium projects and microservices."
    },
    {
        "category": "Programming",
        "question": "How do I learn programming?",
        "answer": "Start with Python - it's beginner-friendly! Use free resources like Codecademy, freeCodeCamp, or YouTube tutorials. Practice daily with small projects, join coding communities, and don't be afraid to make mistakes. Consistency is key!"
    },
    {
        "category": "Programming",
        "question": "What is an API?",
        "answer": "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate. For example, when you use a weather app, it uses an API to fetch data from a weather service. APIs power most modern software integrations."
    },
    
    # Career & Learning
    {
        "category": "Career",
        "question": "How do I start a career in AI?",
        "answer": "Start by learning Python and mathematics (linear algebra, calculus, statistics). Then learn ML/DL frameworks like TensorFlow or PyTorch. Build projects, contribute to open source, and get certifications. Platforms like Coursera, Fast.ai, and Kaggle are great resources!"
    },
    {
        "category": "Career",
        "question": "What skills do I need for data science?",
        "answer": "Key skills include: Python/R programming, statistics and probability, machine learning, data visualization, SQL databases, and domain knowledge. Soft skills like communication and problem-solving are equally important for explaining insights to stakeholders."
    },
    {
        "category": "Career",
        "question": "What is the difference between AI engineer and data scientist?",
        "answer": "AI Engineers focus on building and deploying AI systems in production, working with ML pipelines and infrastructure. Data Scientists focus on analyzing data, building models, and extracting insights. There's overlap, but AI Engineers are more engineering-focused while Data Scientists are more analysis-focused."
    },
    
    # Chatbot Specific
    {
        "category": "Chatbot",
        "question": "What's the difference between rule-based and AI chatbots?",
        "answer": "Rule-based chatbots use predefined patterns and keywords to match responses - they're simple but rigid. AI chatbots (like this one) use machine learning to understand meaning semantically. They can handle variations in phrasing and are more flexible, but require more computational resources."
    },
    {
        "category": "Chatbot",
        "question": "What is cosine similarity?",
        "answer": "Cosine similarity measures the angle between two vectors, ranging from -1 to 1. A value of 1 means identical direction (very similar), 0 means orthogonal (unrelated), and -1 means opposite. It's commonly used to compare text embeddings because it focuses on direction, not magnitude."
    },
    {
        "category": "Chatbot",
        "question": "What is Sentence Transformers?",
        "answer": "Sentence Transformers is a Python library that provides pre-trained models for generating sentence embeddings. It's built on top of Hugging Face Transformers and makes it easy to convert sentences into meaningful vector representations for semantic search, clustering, and similarity tasks."
    },
    
    # Fun & Misc
    {
        "category": "Fun",
        "question": "Tell me a joke about AI",
        "answer": "Why did the neural network break up with the random forest? Because it wanted a deeper connection! 🤖😄"
    },
    {
        "category": "Fun",
        "question": "What's a fun fact about computers?",
        "answer": "The first computer bug was an actual bug! In 1947, Grace Hopper found a moth stuck in a Harvard Mark II computer. She taped it in her logbook and wrote 'First actual case of bug being found.' That's where the term 'debugging' comes from! 🐛"
    },
    {
        "category": "Fun",
        "question": "Who invented the internet?",
        "answer": "The internet wasn't invented by one person! It evolved from ARPANET (1969), created by the US Department of Defense. Key contributors include Vint Cerf and Bob Kahn (TCP/IP protocol), and Tim Berners-Lee (World Wide Web in 1989). It's a collaborative achievement! 🌐"
    },
    
    # 100 Days of AI
    {
        "category": "100 Days of AI",
        "question": "What is the 100 Days of AI challenge?",
        "answer": "The 100 Days of AI is a learning challenge where you build one AI project every day for 100 days. It's a great way to build practical skills, create a portfolio, and stay consistent with learning. This chatbot is Day 47 of the challenge!"
    },
    {
        "category": "100 Days of AI",
        "question": "What projects are in the 100 Days of AI?",
        "answer": "The challenge covers various AI topics: data preprocessing, ML algorithms (regression, classification, clustering), NLP (sentiment analysis, text generation), computer vision, chatbots, and more. Each day builds on previous knowledge while introducing new concepts!"
    },
    
    # Support
    {
        "category": "Support",
        "question": "How can I contact support?",
        "answer": "This is a demo project from the 100 Days of AI challenge. For questions about the code, you can check the GitHub repository or reach out through the project's documentation. Happy learning! 📚"
    },
    {
        "category": "Support",
        "question": "Is this chatbot free to use?",
        "answer": "Yes! This is an open-source educational project. You can use, modify, and learn from the code. It's part of the 100 Days of AI challenge and meant to help others learn about AI and NLP."
    },
    {
        "category": "Support",
        "question": "Can I add my own FAQs?",
        "answer": "Absolutely! The FAQ data is stored in a Python list in app.py. You can easily add new questions and answers by following the same format. The embeddings will be automatically generated when you restart the application."
    }
]

# ===============================
# EMBEDDING ENGINE
# ===============================

class FAQEngine:
    def __init__(self, model_name=MODEL_NAME):
        self.model = None
        self.model_name = model_name
        self.faq_embeddings = None
        self.faqs = FAQ_DATA
        
    def load_model(self):
        """Load the Sentence Transformer model"""
        print(f"📦 Loading Sentence Transformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model loaded successfully!")
        
    def generate_embeddings(self):
        """Generate embeddings for all FAQ questions"""
        if self.model is None:
            self.load_model()
            
        print("🔄 Generating embeddings for FAQ questions...")
        questions = [faq["question"] for faq in self.faqs]
        self.faq_embeddings = self.model.encode(questions, convert_to_numpy=True)
        print(f"✅ Generated {len(self.faq_embeddings)} embeddings!")
        
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def find_best_match(self, query, top_k=3):
        """Find the best matching FAQ for a query"""
        if self.model is None or self.faq_embeddings is None:
            self.load_model()
            self.generate_embeddings()
        
        # Generate embedding for the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities with all FAQ questions
        similarities = []
        for i, faq_embedding in enumerate(self.faq_embeddings):
            similarity = self.cosine_similarity(query_embedding, faq_embedding)
            similarities.append({
                "index": i,
                "similarity": float(similarity),
                "question": self.faqs[i]["question"],
                "answer": self.faqs[i]["answer"],
                "category": self.faqs[i]["category"]
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get top matches
        top_matches = similarities[:top_k]
        
        return top_matches
    
    def get_response(self, query):
        """Get the best response for a user query"""
        if not query or len(query.strip()) < 2:
            return {
                "success": False,
                "error": "Please enter a valid question."
            }
        
        matches = self.find_best_match(query, top_k=3)
        best_match = matches[0]
        
        # Check if the best match is good enough
        if best_match["similarity"] < SIMILARITY_THRESHOLD:
            return {
                "success": True,
                "answer": "I'm not sure I have an answer for that. Could you try rephrasing your question or ask something about AI, programming, or chatbots?",
                "confidence": best_match["similarity"],
                "matched_question": None,
                "category": None,
                "suggestions": [m["question"] for m in matches[:3]]
            }
        
        return {
            "success": True,
            "answer": best_match["answer"],
            "confidence": best_match["similarity"],
            "matched_question": best_match["question"],
            "category": best_match["category"],
            "suggestions": [m["question"] for m in matches[1:3]] if len(matches) > 1 else []
        }
    
    def get_all_faqs(self):
        """Get all FAQs grouped by category"""
        categories = {}
        for faq in self.faqs:
            cat = faq["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "question": faq["question"],
                "answer": faq["answer"]
            })
        return categories
    
    def get_stats(self):
        """Get FAQ engine statistics"""
        categories = {}
        for faq in self.faqs:
            cat = faq["category"]
            categories[cat] = categories.get(cat, 0) + 1
            
        return {
            "total_faqs": len(self.faqs),
            "categories": categories,
            "model": self.model_name,
            "embedding_dimension": 384,  # MiniLM produces 384-dim embeddings
            "threshold": SIMILARITY_THRESHOLD
        }


# Initialize FAQ Engine
faq_engine = FAQEngine()


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Please enter a question'
            }), 400
        
        # Get response from FAQ engine
        response = faq_engine.get_response(query)
        response['timestamp'] = datetime.now().strftime("%I:%M %p")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/faqs', methods=['GET'])
def get_faqs():
    """Get all FAQs"""
    return jsonify({
        'success': True,
        'faqs': faq_engine.get_all_faqs()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get FAQ engine statistics"""
    return jsonify({
        'success': True,
        'stats': faq_engine.get_stats()
    })


@app.route('/search', methods=['POST'])
def search():
    """Search FAQs and return top matches with scores"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Please provide a search query'
            }), 400
        
        matches = faq_engine.find_best_match(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'matches': matches
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===============================
# MAIN
# ===============================

if __name__ == '__main__':
    print("🤖 FAQ Chatbot with Embeddings - Day 47")
    print("=" * 40)
    
    # Pre-load model and generate embeddings
    print("🚀 Initializing FAQ Engine...")
    faq_engine.load_model()
    faq_engine.generate_embeddings()
    
    stats = faq_engine.get_stats()
    print(f"📊 Loaded {stats['total_faqs']} FAQs in {len(stats['categories'])} categories")
    print(f"🧠 Model: {stats['model']}")
    print(f"📐 Embedding dimension: {stats['embedding_dimension']}")
    print("=" * 40)
    print("🚀 Starting server...")
    
    app.run(debug=True, port=5000)
