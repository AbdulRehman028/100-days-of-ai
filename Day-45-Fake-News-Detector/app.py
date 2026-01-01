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
    # Text quality/credibility analysis (using NLI for reliability assessment)
    "credibility": {
        "name": "facebook/bart-large-mnli",
        "display_name": "Credibility Analyzer",
        "description": "Assess text credibility and reliability"
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
credibility_pipeline = None
sentiment_pipeline = None


def load_models():
    """Load all analysis models"""
    global nli_pipeline, credibility_pipeline, sentiment_pipeline
    
    print("📦 Loading NLI model for claim verification...")
    nli_pipeline = pipeline(
        "zero-shot-classification",
        model=MODELS["nli"]["name"],
        device=-1
    )
    print("✅ NLI model loaded!")
    
    # Credibility uses the same NLI model
    credibility_pipeline = nli_pipeline
    print("✅ Credibility analyzer ready!")
    
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
    
    # Truncate long text to avoid model issues
    claim_text = claim[:1024] if len(claim) > 1024 else claim
    
    # Use style-focused labels to avoid confusing topic with quality
    # These labels focus on HOW the content is written, not WHAT it's about
    labels = [
        "This is a well-written, informative article with clear explanations",
        "This is professionally written with proper structure and sources",
        "This uses emotional manipulation or fear-mongering tactics",
        "This makes extraordinary claims without credible evidence",
        "This is balanced reporting that presents multiple perspectives",
        "This is clickbait designed to generate outrage or shock"
    ]
    
    try:
        result = nli_pipeline(claim_text, labels, multi_label=True)
        
        # Map results to meaningful scores
        scores_dict = dict(zip(result["labels"], result["scores"]))
        
        informative_score = scores_dict.get("This is a well-written, informative article with clear explanations", 0)
        professional_score = scores_dict.get("This is professionally written with proper structure and sources", 0)
        manipulation_score = scores_dict.get("This uses emotional manipulation or fear-mongering tactics", 0)
        extraordinary_score = scores_dict.get("This makes extraordinary claims without credible evidence", 0)
        balanced_score = scores_dict.get("This is balanced reporting that presents multiple perspectives", 0)
        clickbait_score = scores_dict.get("This is clickbait designed to generate outrage or shock", 0)
        
        # Calculate combined credibility - positive indicators
        credibility_positive = (informative_score + professional_score + balanced_score) / 3
        # Negative indicators
        credibility_negative = (manipulation_score + extraordinary_score + clickbait_score) / 3
        
        return {
            "factual_score": informative_score,
            "journalism_score": professional_score,
            "balanced_score": balanced_score,
            "misleading_score": manipulation_score,
            "sensational_score": clickbait_score,
            "extraordinary_score": extraordinary_score,
            "opinion_score": 0,  # Not used in new approach
            "promotional_score": 0,  # Not used in new approach
            "credibility_positive": credibility_positive,
            "credibility_negative": credibility_negative,
            "labels": result["labels"],
            "scores": result["scores"]
        }
    except Exception as e:
        print(f"NLI analysis error: {e}")
        return None


def analyze_text_credibility(text):
    """Analyze text credibility using writing quality indicators"""
    global credibility_pipeline
    
    if credibility_pipeline is None:
        return None
    
    # Truncate for model
    text_sample = text[:512] if len(text) > 512 else text
    
    # Check for credibility indicators
    credibility_labels = [
        "professional, academic, or journalistic writing style",
        "informal, emotional, or biased writing style",
        "contains verifiable facts and data",
        "makes unsubstantiated claims"
    ]
    
    try:
        result = credibility_pipeline(text_sample, credibility_labels, multi_label=True)
        scores_dict = dict(zip(result["labels"], result["scores"]))
        
        professional = scores_dict.get("professional, academic, or journalistic writing style", 0)
        informal = scores_dict.get("informal, emotional, or biased writing style", 0)
        verifiable = scores_dict.get("contains verifiable facts and data", 0)
        unsubstantiated = scores_dict.get("makes unsubstantiated claims", 0)
        
        # Calculate overall credibility
        positive_indicators = (professional + verifiable) / 2
        negative_indicators = (informal + unsubstantiated) / 2
        
        return {
            "professional_style": professional,
            "informal_style": informal,
            "verifiable_facts": verifiable,
            "unsubstantiated_claims": unsubstantiated,
            "classification": "CREDIBLE" if positive_indicators > negative_indicators else "QUESTIONABLE",
            "confidence": abs(positive_indicators - negative_indicators),
            "positive_score": positive_indicators,
            "negative_score": negative_indicators
        }
    except Exception as e:
        print(f"Credibility analysis error: {e}")
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


def calculate_verdict(nli_result, credibility_result, red_flags, known_facts, sentiment_result):
    """Calculate final verdict based on all analyses"""
    
    # Initialize scores
    credibility_score = 0.5  # Start neutral
    confidence = 0.0
    reasons = []
    
    # Factor 1: NLI Analysis (Primary - 50% weight)
    if nli_result:
        informative = nli_result.get("factual_score", 0)
        professional = nli_result.get("journalism_score", 0)
        balanced = nli_result.get("balanced_score", 0)
        manipulation = nli_result.get("misleading_score", 0)
        clickbait = nli_result.get("sensational_score", 0)
        extraordinary = nli_result.get("extraordinary_score", 0)
        
        # Positive contributions (good indicators)
        positive_score = (informative + professional + balanced) / 3
        # Negative contributions (bad indicators)
        negative_score = (manipulation + clickbait + extraordinary) / 3
        
        # Calculate net contribution (range -0.25 to +0.25)
        nli_contribution = (positive_score * 0.5) - (negative_score * 0.5)
        credibility_score += nli_contribution
        confidence += 0.4
        
        # Add reasons based on strongest signals
        if informative > 0.25:
            reasons.append(f"Well-written and informative content ({informative:.0%})")
        if professional > 0.25:
            reasons.append(f"Professional writing structure ({professional:.0%})")
        if balanced > 0.25:
            reasons.append(f"Balanced presentation of information ({balanced:.0%})")
        if manipulation > 0.35:
            reasons.append(f"⚠️ May use emotional manipulation ({manipulation:.0%})")
        if clickbait > 0.35:
            reasons.append(f"⚠️ Contains clickbait elements ({clickbait:.0%})")
        if extraordinary > 0.35:
            reasons.append(f"⚠️ Makes extraordinary claims ({extraordinary:.0%})")
    
    # Factor 2: Credibility Analysis (25% weight)
    if credibility_result:
        prof_style = credibility_result.get("professional_style", 0)
        verifiable = credibility_result.get("verifiable_facts", 0)
        
        # Professional style is a strong positive indicator
        if prof_style > 0.5:
            credibility_score += 0.1
            reasons.append(f"Professional writing style ({prof_style:.0%})")
        
        if verifiable > 0.3:
            credibility_score += 0.05
            reasons.append(f"Contains verifiable information ({verifiable:.0%})")
        
        confidence += 0.2
    
    # Factor 3: Red Flags (15% weight) - Only penalize if found
    if red_flags["count"] > 0:
        # Light penalty for red flags, but don't overweight
        penalty = min(red_flags["suspicion_score"] * 0.1, 0.1)
        credibility_score -= penalty
        flag_list = ', '.join(red_flags['flags'][:3])
        reasons.append(f"⚠️ Style concerns: {flag_list}")
        confidence += 0.15
    else:
        credibility_score += 0.05
        reasons.append("No concerning language patterns detected")
        confidence += 0.1
    
    # Factor 4: Sentiment Analysis (10% weight)
    if sentiment_result:
        sentiment = sentiment_result.get("sentiment", "").lower()
        sent_confidence = sentiment_result.get("confidence", 0)
        
        # Neutral is ideal for factual content
        if sentiment == "neutral":
            credibility_score += 0.05
            reasons.append(f"Neutral, objective tone ({sent_confidence:.0%})")
        elif sentiment == "positive":
            # Slight concern for overly positive (could be promotional)
            reasons.append(f"Positive tone detected ({sent_confidence:.0%})")
        elif sentiment == "negative" and sent_confidence > 0.85:
            credibility_score -= 0.03
            reasons.append(f"Strong negative tone ({sent_confidence:.0%})")
        confidence += 0.1
    
    # Factor 5: Known Facts Match (bonus)
    if known_facts:
        reasons.append(f"Relates to known topic: {known_facts[0]['topic']}")
        confidence += 0.05
    
    # Normalize credibility score to 0-1
    credibility_score = max(0, min(1, credibility_score))
    confidence = min(confidence, 1.0)
    
    # Determine verdict based on credibility score
    if confidence < 0.25:
        verdict_key = "UNVERIFIABLE"
    elif credibility_score >= 0.60:
        verdict_key = "TRUE"
    elif credibility_score >= 0.52:
        verdict_key = "MOSTLY_TRUE"
    elif credibility_score >= 0.45:
        verdict_key = "MIXED"
    elif credibility_score >= 0.35:
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
    credibility_result = analyze_text_credibility(claim)
    sentiment_result = analyze_sentiment_bias(claim)
    
    # Calculate verdict with all analysis results
    verdict_result = calculate_verdict(nli_result, credibility_result, red_flags, known_facts, sentiment_result)
    
    return {
        "claim": claim[:500] + "..." if len(claim) > 500 else claim,
        "verdict": verdict_result,
        "analysis": {
            "nli": nli_result,
            "credibility": credibility_result,
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
        'credibility_loaded': credibility_pipeline is not None,
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
