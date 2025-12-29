"""
Day 44: Text Summarizer (Abstractive with BART/T5)
Use BART or T5 for abstractive summarization with ROUGE evaluation
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
# Available models for summarization
MODELS = {
    "bart": {
        "name": "facebook/bart-large-cnn",
        "display_name": "BART Large CNN",
        "description": "Fine-tuned on CNN/DailyMail for news summarization",
        "max_input": 1024,
        "icon": "📰"
    },
    "t5": {
        "name": "t5-small",
        "display_name": "T5 Small",
        "description": "Versatile text-to-text model, good for general summarization",
        "max_input": 512,
        "icon": "🔤"
    },
    "t5-base": {
        "name": "t5-base",
        "display_name": "T5 Base",
        "description": "Larger T5 model for better quality summaries",
        "max_input": 512,
        "icon": "📝"
    },
    "distilbart": {
        "name": "sshleifer/distilbart-cnn-12-6",
        "display_name": "DistilBART CNN",
        "description": "Faster, smaller BART model with good performance",
        "max_input": 1024,
        "icon": "⚡"
    }
}

DEFAULT_MODEL = "distilbart"  # Faster for demo

# Global summarizer
summarizer = None
current_model = None


def load_model(model_key="distilbart"):
    """Load summarization model"""
    global summarizer, current_model
    
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL
    
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    
    # Skip if already loaded
    if current_model == model_key and summarizer is not None:
        return summarizer
    
    print(f"📦 Loading summarization model: {model_info['display_name']}...")
    
    # T5 requires special prefix
    if "t5" in model_key:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU
        )
    else:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1  # CPU
        )
    
    current_model = model_key
    print(f"✅ {model_info['display_name']} loaded successfully!")
    return summarizer


def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores for summary evaluation
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap  
    ROUGE-L: Longest common subsequence
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        return {
            'rouge1': {
                'precision': round(scores['rouge1'].precision * 100, 2),
                'recall': round(scores['rouge1'].recall * 100, 2),
                'f1': round(scores['rouge1'].fmeasure * 100, 2)
            },
            'rouge2': {
                'precision': round(scores['rouge2'].precision * 100, 2),
                'recall': round(scores['rouge2'].recall * 100, 2),
                'f1': round(scores['rouge2'].fmeasure * 100, 2)
            },
            'rougeL': {
                'precision': round(scores['rougeL'].precision * 100, 2),
                'recall': round(scores['rougeL'].recall * 100, 2),
                'f1': round(scores['rougeL'].fmeasure * 100, 2)
            }
        }
    except ImportError:
        return None
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return None


def summarize_text(text, model_key="distilbart", min_length=30, max_length=150, num_beams=4):
    """
    Generate abstractive summary using BART/T5
    
    Args:
        text: Input text to summarize
        model_key: Which model to use
        min_length: Minimum summary length
        max_length: Maximum summary length
        num_beams: Beam search width for better quality
    
    Returns:
        Summary text and metadata
    """
    global summarizer
    
    if summarizer is None or current_model != model_key:
        load_model(model_key)
    
    # Clean and prepare text
    text = text.strip()
    if not text:
        return None
    
    # Get model info
    model_info = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    
    # For T5, add prefix
    if "t5" in model_key:
        input_text = "summarize: " + text
    else:
        input_text = text
    
    try:
        # Generate summary
        result = summarizer(
            input_text,
            min_length=min_length,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )
        
        summary = result[0]['summary_text']
        
        # Calculate compression ratio
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = round((1 - summary_words / original_words) * 100, 1) if original_words > 0 else 0
        
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': compression_ratio,
            'model_used': model_info['display_name']
        }
        
    except Exception as e:
        print(f"Summarization error: {e}")
        return None


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize text endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_key = data.get('model', DEFAULT_MODEL)
        min_length = int(data.get('min_length', 30))
        max_length = int(data.get('max_length', 150))
        reference_summary = data.get('reference_summary', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'Please provide text to summarize'}), 400
        
        if len(text.split()) < 20:
            return jsonify({'success': False, 'error': 'Text too short. Please provide at least 20 words.'}), 400
        
        # Generate summary
        result = summarize_text(text, model_key, min_length, max_length)
        
        if not result:
            return jsonify({'success': False, 'error': 'Could not generate summary'}), 500
        
        # Calculate ROUGE if reference provided
        rouge_scores = None
        if reference_summary:
            rouge_scores = calculate_rouge(reference_summary, result['summary'])
        
        return jsonify({
            'success': True,
            'summary': result['summary'],
            'stats': {
                'original_length': result['original_length'],
                'summary_length': result['summary_length'],
                'original_words': result['original_words'],
                'summary_words': result['summary_words'],
                'compression_ratio': result['compression_ratio'],
                'model_used': result['model_used']
            },
            'rouge_scores': rouge_scores
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    models_list = []
    for key, info in MODELS.items():
        models_list.append({
            'key': key,
            'name': info['display_name'],
            'description': info['description'],
            'icon': info['icon'],
            'max_input': info['max_input']
        })
    return jsonify({
        'models': models_list,
        'current_model': current_model,
        'default_model': DEFAULT_MODEL
    })


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    return jsonify({
        'model_loaded': summarizer is not None,
        'current_model': current_model,
        'model_name': MODELS.get(current_model, {}).get('display_name', 'None')
    })


@app.route('/sample-articles', methods=['GET'])
def sample_articles():
    """Get sample articles for testing"""
    samples = [
        {
            "title": "Climate Change Impact",
            "text": """Climate change is one of the most pressing challenges facing our planet today. Rising global temperatures are causing widespread environmental changes, including melting ice caps, rising sea levels, and more frequent extreme weather events. Scientists warn that without significant reductions in greenhouse gas emissions, these effects will intensify in the coming decades. The Paris Agreement, signed by nearly 200 countries, aims to limit global warming to 1.5 degrees Celsius above pre-industrial levels. However, many experts argue that current commitments are insufficient to meet this goal. Renewable energy sources like solar and wind power are becoming increasingly cost-competitive with fossil fuels, offering hope for a transition to a low-carbon economy. Governments, businesses, and individuals all have roles to play in addressing this global crisis through policy changes, technological innovation, and lifestyle modifications.""",
            "reference": "Climate change poses a major threat with rising temperatures causing environmental damage. The Paris Agreement aims to limit warming to 1.5°C, but current efforts may be insufficient. Renewable energy offers hope for reducing emissions."
        },
        {
            "title": "Artificial Intelligence Revolution",
            "text": """Artificial intelligence has made remarkable progress in recent years, transforming industries from healthcare to transportation. Machine learning algorithms can now diagnose diseases, drive cars, and even create art and music. Large language models like GPT have demonstrated unprecedented capabilities in understanding and generating human-like text. However, these advances also raise important ethical questions about privacy, job displacement, and the potential risks of increasingly powerful AI systems. Tech companies are investing billions of dollars in AI research, while governments are working to develop regulatory frameworks to ensure responsible development. Experts predict that AI will continue to evolve rapidly, with potential applications ranging from personalized medicine to climate modeling. The challenge lies in harnessing these powerful tools while mitigating their risks and ensuring their benefits are widely shared.""",
            "reference": "AI has advanced significantly, transforming healthcare, transportation, and creative fields. While offering great potential, it raises ethical concerns about privacy and job displacement. Balancing innovation with responsible development remains crucial."
        },
        {
            "title": "Space Exploration Milestones",
            "text": """The past decade has witnessed remarkable achievements in space exploration. NASA's Perseverance rover successfully landed on Mars and has been collecting samples that may reveal evidence of ancient microbial life. Private companies like SpaceX have revolutionized the industry with reusable rockets, dramatically reducing the cost of reaching orbit. The James Webb Space Telescope, launched in 2021, is providing unprecedented views of distant galaxies and exoplanets. Meanwhile, China has established its own space station and landed rovers on the far side of the Moon. Plans are underway for crewed missions to Mars within the next two decades, with both NASA and SpaceX developing the necessary technologies. These endeavors not only expand our understanding of the universe but also drive technological innovations that benefit life on Earth.""",
            "reference": "Recent space exploration includes Mars rovers, reusable rockets, and the James Webb Telescope. Multiple nations and private companies are advancing capabilities, with Mars missions planned for the coming decades."
        }
    ]
    return jsonify({'samples': samples})


# ===============================
# INITIALIZATION
# ===============================

print("📝 Text Summarizer - Day 44")
print("=" * 35)


if __name__ == '__main__':
    # Pre-load default model at startup
    print("🚀 Pre-loading summarization model...")
    load_model(DEFAULT_MODEL)
    print("✅ Ready to summarize!")
    app.run(debug=True, port=5000)
