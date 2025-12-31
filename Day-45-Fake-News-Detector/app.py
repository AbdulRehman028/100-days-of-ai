"""
Day 45: Fake News Detector (LLM Fact-Checking)
Use LLM to analyze claims, check facts, and output verdicts
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================

# Models for different analysis tasks
MODELS = {
    # Natural Language Inference for claim verification
    "nli": {
        "name": "facebook/bart-large-mnli",
        "display_name": "BART MNLI",
        "description": "Natural Language Inference for claim verification"
    },
    # Fake news detection model
    "fake_news": {
        "name": "hamzab/roberta-fake-news-classification",
        "display_name": "RoBERTa Fake News",
        "description": "Fine-tuned for fake news classification"
    },
    # Sentiment/Bias analysis
    "sentiment": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "display_name": "RoBERTa Sentiment",
        "description": "Detect emotional bias in text"
    }
}

# Verdict categories
VERDICTS = {
    "TRUE": {
        "label": "True",
        "icon": "✅",
        "color": "#22c55e",
        "description": "The claim appears to be factually accurate"
    },
    "MOSTLY_TRUE": {
        "label": "Mostly True",
        "icon": "🟢",
        "color": "#84cc16",
        "description": "The claim is largely accurate with minor issues"
    },
    "MIXED": {
        "label": "Mixed",
        "icon": "🟡",
        "color": "#eab308",
        "description": "The claim contains both true and false elements"
    },
    "MOSTLY_FALSE": {
        "label": "Mostly False",
        "icon": "🟠",
        "color": "#f97316",
        "description": "The claim is largely inaccurate"
    },
    "FALSE": {
        "label": "False",
        "icon": "❌",
        "color": "#ef4444",
        "description": "The claim appears to be factually incorrect"
    },
    "UNVERIFIABLE": {
        "label": "Unverifiable",
        "icon": "❓",
        "color": "#6b7280",
        "description": "Cannot determine accuracy without more context"
    }
}

# Common fact-checking knowledge base (simplified)
KNOWN_FACTS = {
    "earth_shape": {
        "fact": "The Earth is an oblate spheroid (roughly spherical)",
        "keywords": ["earth", "flat", "round", "sphere", "globe"]
    },
    "climate_change": {
        "fact": "Climate change is supported by scientific consensus",
        "keywords": ["climate", "global warming", "hoax", "scientists"]
    },
    "vaccines": {
        "fact": "Vaccines are safe and effective according to medical consensus",
        "keywords": ["vaccine", "autism", "dangerous", "safe"]
    },
    "moon_landing": {
        "fact": "NASA landed astronauts on the Moon in 1969",
        "keywords": ["moon", "landing", "fake", "nasa", "1969"]
    },
    "evolution": {
        "fact": "Evolution is the scientific consensus for biodiversity",
        "keywords": ["evolution", "darwin", "species", "creation"]
    }
}

# Red flag indicators for fake news
RED_FLAGS = [
    {"pattern": r"\b(SHOCKING|BREAKING|URGENT|BOMBSHELL)\b", "flag": "Sensationalist language", "weight": 0.15},
    {"pattern": r"\b(they don't want you to know|secret|hidden truth)\b", "flag": "Conspiracy language", "weight": 0.2},
    {"pattern": r"\b(100%|guaranteed|proven|definitely|always|never)\b", "flag": "Absolute claims", "weight": 0.1},
    {"pattern": r"\b(mainstream media|MSM|fake news media)\b", "flag": "Media distrust language", "weight": 0.15},
    {"pattern": r"\b(miracle|cure-all|wonder)\b", "flag": "Miracle claims", "weight": 0.15},
    {"pattern": r"\b(exposed|cover-up|conspiracy)\b", "flag": "Conspiracy terminology", "weight": 0.15},
    {"pattern": r"!!+|\?\?+", "flag": "Excessive punctuation", "weight": 0.1},
    {"pattern": r"\b[A-Z]{4,}\b", "flag": "Excessive capitalization", "weight": 0.1},
]

# Global pipelines
nli_pipeline = None
fake_news_pipeline = None
sentiment_pipeline = None


def load_models():
    """Load all analysis models"""
    global nli_pipeline, fake_news_pipeline, sentiment_pipeline
    
    print("📦 Loading NLI model for claim verification...")
    nli_pipeline = pipeline(
        "zero-shot-classification",
        model=MODELS["nli"]["name"],
        device=-1
    )
    print("✅ NLI model loaded!")
    
    print("📦 Loading Fake News detection model...")
    try:
        fake_news_pipeline = pipeline(
            "text-classification",
            model=MODELS["fake_news"]["name"],
            device=-1
        )
        print("✅ Fake News model loaded!")
    except Exception as e:
        print(f"⚠️ Fake news model not available: {e}")
        fake_news_pipeline = None
    
    print("📦 Loading Sentiment analysis model...")
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=MODELS["sentiment"]["name"],
            device=-1
        )
        print("✅ Sentiment model loaded!")
    except Exception as e:
        print(f"⚠️ Sentiment model not available: {e}")
        sentiment_pipeline = None


def analyze_red_flags(text):
    """Detect red flags in the text"""
    flags_found = []
    total_weight = 0
    
    text_lower = text.lower()
    
    for flag_info in RED_FLAGS:
        if re.search(flag_info["pattern"], text, re.IGNORECASE):
            flags_found.append(flag_info["flag"])
            total_weight += flag_info["weight"]
    
    return {
        "flags": flags_found,
        "count": len(flags_found),
        "suspicion_score": min(total_weight, 1.0)  # Cap at 1.0
    }


def check_against_known_facts(claim):
    """Check claim against known facts database"""
    claim_lower = claim.lower()
    
    relevant_facts = []
    for fact_key, fact_info in KNOWN_FACTS.items():
        # Check if any keywords match
        matches = sum(1 for kw in fact_info["keywords"] if kw in claim_lower)
        if matches >= 2:  # At least 2 keyword matches
            relevant_facts.append({
                "topic": fact_key.replace("_", " ").title(),
                "established_fact": fact_info["fact"],
                "relevance": matches
            })
    
    return sorted(relevant_facts, key=lambda x: x["relevance"], reverse=True)


def analyze_claim_with_nli(claim, evidence=None):
    """Use NLI to verify claim against evidence"""
    global nli_pipeline
    
    if nli_pipeline is None:
        return None
    
    # If no evidence provided, use general fact-checking labels
    if not evidence:
        labels = [
            "This is a factual and accurate statement",
            "This is misleading or contains false information",
            "This is an opinion or cannot be verified",
            "This contains exaggerated or sensationalized claims"
        ]
        
        result = nli_pipeline(claim, labels, multi_label=True)
        
        return {
            "factual_score": result["scores"][0],
            "misleading_score": result["scores"][1],
            "opinion_score": result["scores"][2],
            "sensational_score": result["scores"][3],
            "labels": result["labels"],
            "scores": result["scores"]
        }
    else:
        # Check claim against provided evidence
        labels = ["entailment", "contradiction", "neutral"]
        hypothesis = f"Based on the evidence: {evidence}"
        
        result = nli_pipeline(
            claim,
            labels,
            hypothesis_template="{}. " + hypothesis
        )
        
        return {
            "supports": result["scores"][0],
            "contradicts": result["scores"][1],
            "neutral": result["scores"][2]
        }


def analyze_fake_news(text):
    """Direct fake news classification"""
    global fake_news_pipeline
    
    if fake_news_pipeline is None:
        return None
    
    try:
        result = fake_news_pipeline(text[:512])  # Truncate for model
        
        # Handle different model output formats
        if isinstance(result, list):
            result = result[0]
        
        label = result.get("label", "").upper()
        score = result.get("score", 0)
        
        # Normalize labels
        if "FAKE" in label or "FALSE" in label or label == "LABEL_0":
            return {"classification": "FAKE", "confidence": score}
        elif "REAL" in label or "TRUE" in label or label == "LABEL_1":
            return {"classification": "REAL", "confidence": score}
        else:
            return {"classification": label, "confidence": score}
            
    except Exception as e:
        print(f"Fake news analysis error: {e}")
        return None


def analyze_sentiment_bias(text):
    """Analyze emotional bias in text"""
    global sentiment_pipeline
    
    if sentiment_pipeline is None:
        return None
    
    try:
        result = sentiment_pipeline(text[:512])
        
        if isinstance(result, list):
            result = result[0]
        
        return {
            "sentiment": result.get("label", "neutral"),
            "confidence": result.get("score", 0)
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return None


def calculate_verdict(nli_result, fake_news_result, red_flags, known_facts):
    """Calculate final verdict based on all analyses"""
    
    # Initialize scores
    credibility_score = 0.5  # Start neutral
    confidence = 0.0
    reasons = []
    
    # Factor 1: NLI Analysis (40% weight)
    if nli_result:
        factual = nli_result.get("factual_score", 0)
        misleading = nli_result.get("misleading_score", 0)
        sensational = nli_result.get("sensational_score", 0)
        
        nli_contribution = (factual * 0.5) - (misleading * 0.3) - (sensational * 0.2)
        credibility_score += nli_contribution * 0.4
        confidence += 0.3
        
        if factual > 0.5:
            reasons.append(f"NLI analysis suggests factual content ({factual:.0%})")
        if misleading > 0.4:
            reasons.append(f"NLI detects potentially misleading elements ({misleading:.0%})")
        if sensational > 0.4:
            reasons.append(f"NLI detects sensationalized language ({sensational:.0%})")
    
    # Factor 2: Fake News Model (30% weight)
    if fake_news_result:
        if fake_news_result["classification"] == "REAL":
            credibility_score += 0.3 * fake_news_result["confidence"]
            reasons.append(f"Fake news detector: REAL ({fake_news_result['confidence']:.0%})")
        elif fake_news_result["classification"] == "FAKE":
            credibility_score -= 0.3 * fake_news_result["confidence"]
            reasons.append(f"Fake news detector: FAKE ({fake_news_result['confidence']:.0%})")
        confidence += 0.25
    
    # Factor 3: Red Flags (20% weight)
    if red_flags["count"] > 0:
        credibility_score -= red_flags["suspicion_score"] * 0.2
        reasons.append(f"Found {red_flags['count']} red flag(s): {', '.join(red_flags['flags'][:3])}")
        confidence += 0.2
    
    # Factor 4: Known Facts Match (10% weight)
    if known_facts:
        # Having relevant established facts to compare against
        reasons.append(f"Related to known topic: {known_facts[0]['topic']}")
        confidence += 0.15
    
    # Normalize credibility score to 0-1
    credibility_score = max(0, min(1, credibility_score))
    confidence = min(confidence, 1.0)
    
    # Determine verdict
    if confidence < 0.3:
        verdict_key = "UNVERIFIABLE"
    elif credibility_score >= 0.7:
        verdict_key = "TRUE"
    elif credibility_score >= 0.55:
        verdict_key = "MOSTLY_TRUE"
    elif credibility_score >= 0.45:
        verdict_key = "MIXED"
    elif credibility_score >= 0.3:
        verdict_key = "MOSTLY_FALSE"
    else:
        verdict_key = "FALSE"
    
    return {
        "verdict": verdict_key,
        "verdict_info": VERDICTS[verdict_key],
        "credibility_score": round(credibility_score * 100, 1),
        "confidence": round(confidence * 100, 1),
        "reasons": reasons
    }


def fact_check_claim(claim, context=None):
    """
    Main fact-checking function
    
    Args:
        claim: The news claim or article to check
        context: Optional additional context
    
    Returns:
        Complete fact-check analysis
    """
    
    if not claim or len(claim.strip()) < 10:
        return {"error": "Please provide a longer claim to analyze"}
    
    # Run all analyses
    red_flags = analyze_red_flags(claim)
    known_facts = check_against_known_facts(claim)
    nli_result = analyze_claim_with_nli(claim)
    fake_news_result = analyze_fake_news(claim)
    sentiment_result = analyze_sentiment_bias(claim)
    
    # Calculate verdict
    verdict_result = calculate_verdict(nli_result, fake_news_result, red_flags, known_facts)
    
    return {
        "claim": claim[:500] + "..." if len(claim) > 500 else claim,
        "verdict": verdict_result,
        "analysis": {
            "nli": nli_result,
            "fake_news": fake_news_result,
            "sentiment": sentiment_result,
            "red_flags": red_flags,
            "known_facts": known_facts
        },
        "word_count": len(claim.split()),
        "char_count": len(claim)
    }


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check_claim():
    """Fact-check a claim"""
    try:
        data = request.get_json()
        claim = data.get('claim', '').strip()
        context = data.get('context', '').strip()
        
        if not claim:
            return jsonify({'success': False, 'error': 'Please provide a claim to check'}), 400
        
        if len(claim) < 20:
            return jsonify({'success': False, 'error': 'Claim too short. Please provide more text.'}), 400
        
        # Run fact-check
        result = fact_check_claim(claim, context if context else None)
        
        if "error" in result:
            return jsonify({'success': False, 'error': result["error"]}), 400
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-check', methods=['POST'])
def batch_check():
    """Check multiple claims"""
    try:
        data = request.get_json()
        claims = data.get('claims', [])
        
        if not claims:
            return jsonify({'success': False, 'error': 'Please provide claims to check'}), 400
        
        results = []
        for claim in claims[:5]:  # Limit to 5 claims
            if claim.strip():
                result = fact_check_claim(claim.strip())
                results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/verdicts', methods=['GET'])
def get_verdicts():
    """Get verdict definitions"""
    return jsonify({'verdicts': VERDICTS})


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    return jsonify({
        'nli_loaded': nli_pipeline is not None,
        'fake_news_loaded': fake_news_pipeline is not None,
        'sentiment_loaded': sentiment_pipeline is not None
    })


@app.route('/sample-claims', methods=['GET'])
def sample_claims():
    """Get sample claims for testing"""
    samples = [
        {
            "title": "Scientific Claim",
            "claim": "Scientists have discovered that drinking coffee every day can extend your lifespan by up to 10 years, according to a new study published in Nature.",
            "category": "health"
        },
        {
            "title": "Conspiracy Theory",
            "claim": "BREAKING: Government documents EXPOSED showing they've been hiding the truth about 5G towers causing health problems. The mainstream media doesn't want you to know this!!!",
            "category": "conspiracy"
        },
        {
            "title": "Political Claim",
            "claim": "The new infrastructure bill will create over 2 million jobs and invest $550 billion in roads, bridges, and broadband internet over the next five years.",
            "category": "politics"
        },
        {
            "title": "Viral Misinformation",
            "claim": "URGENT: Eating bananas and drinking warm water can cure any virus within 24 hours. Doctors are keeping this secret because it would destroy the pharmaceutical industry!",
            "category": "health"
        },
        {
            "title": "Factual News",
            "claim": "The James Webb Space Telescope has captured new images of distant galaxies, providing scientists with unprecedented data about the early universe.",
            "category": "science"
        },
        {
            "title": "Exaggerated Claim",
            "claim": "This miracle supplement has been 100% PROVEN to make you lose 30 pounds in just one week with NO exercise and NO diet changes. Guaranteed results!",
            "category": "health"
        }
    ]
    return jsonify({'samples': samples})


# ===============================
# INITIALIZATION
# ===============================

print("🔍 Fake News Detector - Day 45")
print("=" * 35)


if __name__ == '__main__':
    # Pre-load models at startup
    print("🚀 Pre-loading fact-checking models...")
    load_models()
    print("✅ Ready to detect fake news!")
    app.run(debug=True, port=5000)
