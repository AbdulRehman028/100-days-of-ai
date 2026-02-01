"""
Day 52: Question-Answering System (Extractive to Generative)
A QA system that supports both extractive and generative approaches.

Extractive QA: Finds and extracts the answer span from the context
Generative QA: Generates a natural language answer based on the context

Tech Stack: Python, Flask, Hugging Face Transformers
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import torch
import time
import re

app = Flask(__name__)

# Banner
print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         Day 52: Question-Answering System                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Extractive: DistilBERT (distilbert-base-uncased-distilled   ║
    ║              -squad)                                         ║
    ║  Generative: BART (facebook/bart-large-cnn) [Cached]         ║
    ║  Features: Dual QA Modes + Confidence Scoring                ║
    ╚══════════════════════════════════════════════════════════════╝
""")


class QuestionAnsweringSystem:
    """Dual-mode Question Answering System"""
    
    def __init__(self):
        self.extractive_pipeline = None
        self.gen_model = None
        self.gen_tokenizer = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        
    def load_extractive_model(self):
        """Load the extractive QA model (DistilBERT fine-tuned on SQuAD)"""
        if self.extractive_pipeline is None:
            print("🔧 Loading Extractive QA model (DistilBERT-SQuAD)...")
            self.extractive_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad",
                device=self.device
            )
            print("✅ Extractive model loaded!")
        return self.extractive_pipeline
    
    def load_generative_model(self):
        """Load the generative QA model using direct model loading"""
        if self.gen_model is None:
            print("🔧 Loading Generative QA model (BART-large-CNN)...")
            print("   📦 Using cached model from Day 51!")
            from transformers import BartForConditionalGeneration, BartTokenizer
            self.gen_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.gen_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            print("✅ Generative model loaded!")
        return self.gen_model
    
    def extractive_qa(self, context: str, question: str) -> dict:
        """
        Extractive QA: Find the exact answer span in the context.
        Returns the extracted text with confidence score.
        """
        pipe = self.load_extractive_model()
        
        start_time = time.time()
        result = pipe(question=question, context=context)
        elapsed_time = time.time() - start_time
        
        # Highlight the answer in context
        answer = result['answer']
        highlighted_context = self._highlight_answer(context, answer)
        
        return {
            'answer': answer,
            'confidence': round(result['score'] * 100, 2),
            'start': result['start'],
            'end': result['end'],
            'highlighted_context': highlighted_context,
            'processing_time': round(elapsed_time, 3),
            'mode': 'extractive',
            'model': 'DistilBERT-SQuAD'
        }
    
    def generative_qa(self, context: str, question: str) -> dict:
        """
        Generative QA: Generate a natural language answer.
        Uses BART to produce fluent responses based on context.
        """
        self.load_generative_model()
        
        # Format the input for BART
        input_text = f"Answer this question: {question}\n\nBased on: {context}"
        
        start_time = time.time()
        
        # Tokenize and generate
        inputs = self.gen_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.gen_model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=20,
            num_beams=4,
            early_stopping=True
        )
        answer = self.gen_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        elapsed_time = time.time() - start_time
        
        # Calculate a pseudo-confidence based on answer relevance
        confidence = self._calculate_relevance_score(context, question, answer)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'processing_time': round(elapsed_time, 3),
            'mode': 'generative',
            'model': 'BART-large-CNN'
        }
    
    def hybrid_qa(self, context: str, question: str) -> dict:
        """
        Hybrid QA: Use both extractive and generative approaches,
        then combine insights from both.
        """
        extractive_result = self.extractive_qa(context, question)
        generative_result = self.generative_qa(context, question)
        
        return {
            'extractive': extractive_result,
            'generative': generative_result,
            'mode': 'hybrid',
            'recommendation': self._get_recommendation(extractive_result, generative_result)
        }
    
    def _highlight_answer(self, context: str, answer: str) -> str:
        """Highlight the answer within the context using HTML markup"""
        # Case-insensitive replacement but preserve original case
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark class="bg-gradient-to-r from-emerald-400 to-cyan-400 text-gray-900 px-1 rounded font-semibold">{answer}</mark>',
            context,
            count=1
        )
        return highlighted
    
    def _calculate_relevance_score(self, context: str, question: str, answer: str) -> float:
        """Calculate a pseudo-relevance score for generative answers"""
        # Simple heuristic: check word overlap between answer and context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        relevance = min(100, (overlap / len(answer_words)) * 100)
        
        # Boost if answer is substantial
        if len(answer.split()) >= 5:
            relevance = min(100, relevance + 10)
        
        return round(relevance, 2)
    
    def _get_recommendation(self, extractive: dict, generative: dict) -> str:
        """Provide a recommendation on which answer to trust"""
        if extractive['confidence'] > 80:
            return "The extractive answer has high confidence and is likely accurate."
        elif extractive['confidence'] > 50:
            return "Consider both answers. The extractive provides exact text, while generative offers a natural response."
        else:
            return "The generative answer may be more helpful as the extractive confidence is low."


# Initialize the QA system
qa_system = QuestionAnsweringSystem()


# Sample contexts for demo
SAMPLE_CONTEXTS = {
    'science': {
        'title': '🔬 Science - Photosynthesis',
        'context': """Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can be stored and later released to fuel the organism's activities. This process takes place primarily in the chloroplasts of plant cells, specifically in structures called thylakoids. During photosynthesis, plants absorb carbon dioxide from the air through small pores called stomata and water from the soil through their roots. Using sunlight as an energy source, they convert these raw materials into glucose (a simple sugar) and oxygen. The overall chemical equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Chlorophyll, the green pigment in plants, plays a crucial role by absorbing light energy, primarily from the blue and red portions of the electromagnetic spectrum."""
    },
    'history': {
        'title': '📜 History - Moon Landing',
        'context': """The Apollo 11 mission was the first crewed mission to land on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, and collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. Command module pilot Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface. Armstrong's first step onto the lunar surface was broadcast on live TV to a worldwide audience, and he described the event as "one small step for man, one giant leap for mankind." """
    },
    'technology': {
        'title': '💻 Technology - Machine Learning',
        'context': """Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms build mathematical models based on training data to make predictions or decisions without being explicitly programmed for the task. There are three main types of machine learning: supervised learning, where the algorithm learns from labeled data; unsupervised learning, where the algorithm finds patterns in unlabeled data; and reinforcement learning, where an agent learns to make decisions by interacting with an environment. Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn hierarchical representations of data. Popular frameworks for machine learning include TensorFlow, PyTorch, and scikit-learn."""
    },
    'literature': {
        'title': '📚 Literature - Shakespeare',
        'context': """William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language and the world's greatest dramatist. He was born in Stratford-upon-Avon in 1564 and died there in 1616. Shakespeare wrote approximately 39 plays, 154 sonnets, and several longer poems. His plays have been translated into every major language and are performed more often than those of any other playwright. His works include famous tragedies such as Hamlet, Macbeth, Othello, and King Lear, as well as beloved comedies like A Midsummer Night's Dream, Much Ado About Nothing, and The Tempest. The Globe Theatre in London, associated with Shakespeare, was built in 1599 by his playing company, the Lord Chamberlain's Men."""
    }
}


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', samples=SAMPLE_CONTEXTS)


@app.route('/ask', methods=['POST'])
def ask():
    """Process a question and return the answer"""
    try:
        data = request.get_json()
        context = data.get('context', '').strip()
        question = data.get('question', '').strip()
        mode = data.get('mode', 'extractive')
        
        # Validation
        if not context:
            return jsonify({'error': 'Please provide a context'}), 400
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        if len(context) < 50:
            return jsonify({'error': 'Context is too short. Please provide more text.'}), 400
        if len(question) < 5:
            return jsonify({'error': 'Question is too short.'}), 400
        
        # Process based on mode
        if mode == 'extractive':
            result = qa_system.extractive_qa(context, question)
        elif mode == 'generative':
            result = qa_system.generative_qa(context, question)
        elif mode == 'hybrid':
            result = qa_system.hybrid_qa(context, question)
        else:
            return jsonify({'error': 'Invalid mode. Use extractive, generative, or hybrid.'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': qa_system.device_name,
        'extractive_loaded': qa_system.extractive_pipeline is not None,
        'generative_loaded': qa_system.gen_model is not None
    })


if __name__ == '__main__':
    print("🚀 Starting Question-Answering System...")
    print(f"📱 Device: {qa_system.device_name}")
    print("=" * 50)
    
    # Pre-load models for faster first response
    print("\n📦 Pre-loading models (this may take a moment)...")
    qa_system.load_extractive_model()
    qa_system.load_generative_model()
    print("\n✨ All models loaded! Starting server...")
    print("=" * 50)
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
