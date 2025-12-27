"""
Day 42: Keyword Extractor with Embeddings
Extract keywords using embeddings and clustering
"""

import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and good quality
MAX_KEYWORDS = 15
MIN_KEYWORD_LENGTH = 3

# Global model
model = None


def load_model():
    """Load Sentence Transformer model"""
    global model
    
    print(f"📦 Loading Sentence Transformer: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded successfully!")
    return model


def preprocess_text(text):
    """Clean and preprocess text"""
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_candidate_keywords(text, ngram_range=(1, 3)):
    """Extract candidate keywords/phrases using n-grams"""
    # Use CountVectorizer to extract n-grams
    try:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words='english',
            max_features=100
        )
        vectorizer.fit([text])
        candidates = vectorizer.get_feature_names_out()
        
        # Filter by length
        candidates = [c for c in candidates if len(c) >= MIN_KEYWORD_LENGTH]
        
        return list(candidates)
    except Exception as e:
        # Fallback: simple word extraction
        words = text.lower().split()
        return list(set([w for w in words if len(w) >= MIN_KEYWORD_LENGTH]))


def extract_keywords_embedding(text, num_keywords=10, diversity=0.5):
    """
    Extract keywords using embedding similarity
    
    Algorithm:
    1. Extract candidate keywords/phrases
    2. Compute embeddings for document and candidates
    3. Rank by similarity to document
    4. Apply MMR for diversity
    """
    global model
    
    if model is None:
        load_model()
    
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Extract candidates
    candidates = extract_candidate_keywords(clean_text)
    
    if not candidates:
        return []
    
    # Compute embeddings
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    
    # Calculate similarity to document
    doc_similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]
    
    # Calculate candidate-candidate similarities for diversity
    candidate_similarities = cosine_similarity(candidate_embeddings)
    
    # Apply Maximal Marginal Relevance (MMR) for diverse keywords
    keywords = mmr_selection(
        doc_similarities,
        candidate_similarities,
        candidates,
        num_keywords,
        diversity
    )
    
    return keywords


def mmr_selection(doc_sim, cand_sim, candidates, num_keywords, diversity):
    """
    Maximal Marginal Relevance selection for diverse keywords
    
    MMR = λ * sim(candidate, doc) - (1-λ) * max(sim(candidate, selected))
    """
    selected_indices = []
    selected_keywords = []
    
    # Get indices sorted by document similarity
    unselected = list(range(len(candidates)))
    
    for _ in range(min(num_keywords, len(candidates))):
        if not unselected:
            break
            
        mmr_scores = []
        
        for idx in unselected:
            # Document relevance
            relevance = doc_sim[idx]
            
            # Diversity penalty (similarity to already selected)
            if selected_indices:
                redundancy = max([cand_sim[idx][s] for s in selected_indices])
            else:
                redundancy = 0
            
            # MMR score
            mmr = diversity * relevance - (1 - diversity) * redundancy
            mmr_scores.append((idx, mmr, relevance))
        
        # Select best MMR score
        best = max(mmr_scores, key=lambda x: x[1])
        best_idx = best[0]
        
        selected_indices.append(best_idx)
        selected_keywords.append({
            'keyword': candidates[best_idx],
            'relevance': float(best[2]),
            'mmr_score': float(best[1])
        })
        unselected.remove(best_idx)
    
    return selected_keywords


def extract_keywords_clustering(text, num_clusters=5):
    """
    Extract keywords using clustering approach
    
    Algorithm:
    1. Extract candidate keywords
    2. Cluster candidates by embedding similarity
    3. Select representative from each cluster
    """
    global model
    
    if model is None:
        load_model()
    
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Extract candidates
    candidates = extract_candidate_keywords(clean_text)
    
    if len(candidates) < num_clusters:
        num_clusters = max(1, len(candidates) // 2)
    
    if not candidates:
        return [], []
    
    # Compute embeddings
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    
    # Cluster candidates
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(candidate_embeddings)
    
    # Calculate document similarities
    doc_similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]
    
    # Select best keyword from each cluster
    keywords = []
    cluster_info = []
    
    for cluster_id in range(num_clusters):
        # Get candidates in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Find most relevant to document in this cluster
        cluster_similarities = [(i, doc_similarities[i]) for i in cluster_indices]
        best_idx, best_sim = max(cluster_similarities, key=lambda x: x[1])
        
        keywords.append({
            'keyword': candidates[best_idx],
            'relevance': float(best_sim),
            'cluster': int(cluster_id)
        })
        
        # Store cluster info
        cluster_members = [candidates[i] for i in cluster_indices]
        cluster_info.append({
            'cluster_id': int(cluster_id),
            'representative': candidates[best_idx],
            'members': cluster_members[:5],  # Top 5 members
            'size': len(cluster_indices)
        })
    
    # Sort by relevance
    keywords.sort(key=lambda x: x['relevance'], reverse=True)
    
    return keywords, cluster_info


def get_keyword_context(text, keyword):
    """Find context snippet where keyword appears"""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    pos = text_lower.find(keyword_lower)
    if pos == -1:
        return None
    
    # Get surrounding context (50 chars each side)
    start = max(0, pos - 50)
    end = min(len(text), pos + len(keyword) + 50)
    
    snippet = text[start:end]
    if start > 0:
        snippet = '...' + snippet
    if end < len(text):
        snippet = snippet + '...'
    
    return snippet


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    """Extract keywords from text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        method = data.get('method', 'embedding')  # 'embedding' or 'clustering'
        num_keywords = int(data.get('num_keywords', 10))
        diversity = float(data.get('diversity', 0.5))
        
        if not text:
            return jsonify({'success': False, 'error': 'Please provide text to analyze'}), 400
        
        if len(text) < 50:
            return jsonify({'success': False, 'error': 'Please provide longer text (at least 50 characters)'}), 400
        
        # Extract keywords
        if method == 'clustering':
            keywords, cluster_info = extract_keywords_clustering(text, num_clusters=num_keywords)
        else:
            keywords = extract_keywords_embedding(text, num_keywords, diversity)
            cluster_info = None
        
        # Add context to keywords
        for kw in keywords:
            kw['context'] = get_keyword_context(text, kw['keyword'])
        
        # Calculate statistics
        word_count = len(text.split())
        char_count = len(text)
        
        return jsonify({
            'success': True,
            'keywords': keywords,
            'cluster_info': cluster_info,
            'method': method,
            'statistics': {
                'word_count': word_count,
                'char_count': char_count,
                'keywords_found': len(keywords)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/sample-texts', methods=['GET'])
def sample_texts():
    """Get sample texts for demo"""
    samples = [
        {
            'title': 'Machine Learning',
            'text': '''Machine learning is a branch of artificial intelligence that focuses on building systems 
            that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning 
            algorithms identify patterns in data and make decisions with minimal human intervention. Deep learning, 
            a subset of machine learning, uses neural networks with multiple layers to process complex patterns. 
            Applications include image recognition, natural language processing, recommendation systems, and 
            autonomous vehicles. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.'''
        },
        {
            'title': 'Climate Change',
            'text': '''Climate change refers to long-term shifts in global temperatures and weather patterns. 
            While natural factors like volcanic eruptions contribute to climate variability, human activities 
            have been the primary driver since the industrial revolution. Burning fossil fuels releases greenhouse 
            gases like carbon dioxide and methane, trapping heat in the atmosphere. Effects include rising sea 
            levels, extreme weather events, melting ice caps, and biodiversity loss. Solutions involve renewable 
            energy adoption, carbon capture technology, sustainable agriculture, and international policy agreements.'''
        },
        {
            'title': 'Blockchain Technology',
            'text': '''Blockchain is a distributed ledger technology that enables secure, transparent, and 
            tamper-resistant record-keeping. Originally developed for Bitcoin cryptocurrency, blockchain has 
            expanded to various applications including smart contracts, supply chain management, digital identity, 
            and decentralized finance. The technology works through consensus mechanisms like proof of work or 
            proof of stake, where network participants validate transactions. Ethereum pioneered programmable 
            blockchains, enabling developers to build decentralized applications and create non-fungible tokens.'''
        },
        {
            'title': 'Space Exploration',
            'text': '''Space exploration has entered a new era with private companies like SpaceX and Blue Origin 
            joining government agencies. NASA's Artemis program aims to return humans to the Moon and establish 
            a permanent lunar presence. Mars missions, including rovers like Perseverance, search for signs of 
            ancient microbial life. The James Webb Space Telescope captures unprecedented images of distant 
            galaxies. Future goals include asteroid mining, space tourism, and eventually human settlements on 
            Mars. International cooperation through the International Space Station demonstrates peaceful space 
            collaboration.'''
        },
        {
            'title': 'Quantum Computing',
            'text': '''Quantum computing harnesses quantum mechanics principles like superposition and entanglement 
            to process information in fundamentally different ways than classical computers. While traditional 
            computers use bits representing 0 or 1, quantum computers use qubits that can exist in multiple states 
            simultaneously. This enables solving certain problems exponentially faster, including cryptography, 
            drug discovery, financial modeling, and optimization problems. Companies like IBM, Google, and 
            startups are racing to achieve quantum supremacy and build practical quantum computers.'''
        }
    ]
    return jsonify({'samples': samples})


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    return jsonify({
        'model_loaded': model is not None,
        'model_name': MODEL_NAME
    })


# ===============================
# INITIALIZATION
# ===============================

print("🔑 Keyword Extractor with Embeddings - Day 42")
print("=" * 45)

try:
    load_model()
except Exception as e:
    print(f"⚠️ Model will be loaded on first request: {e}")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
