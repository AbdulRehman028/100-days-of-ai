"""
Day 41: Named Entity Recognition (spaCy + LLM Integration)
Use spaCy for NER, enhance with LLM for context-aware entity extraction
"""

import os
import json
import spacy
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
SPACY_MODEL = 'en_core_web_sm'  # Options: en_core_web_sm, en_core_web_md, en_core_web_lg
USE_LLM_ENHANCEMENT = True  # Enable/disable LLM context enhancement

# Global models
nlp = None
llm_classifier = None

# Entity color mapping for visualization
ENTITY_COLORS = {
    'PERSON': '#FF6B6B',      # Red
    'ORG': '#4ECDC4',         # Teal
    'GPE': '#45B7D1',         # Blue (Geo-Political Entity)
    'LOC': '#96CEB4',         # Green (Location)
    'DATE': '#FFEAA7',        # Yellow
    'TIME': '#DDA0DD',        # Plum
    'MONEY': '#98D8C8',       # Mint
    'PERCENT': '#F7DC6F',     # Gold
    'PRODUCT': '#BB8FCE',     # Purple
    'EVENT': '#F1948A',       # Salmon
    'WORK_OF_ART': '#85C1E9', # Sky Blue
    'LAW': '#F8B500',         # Orange
    'LANGUAGE': '#58D68D',    # Light Green
    'FAC': '#AF7AC5',         # Facility - Purple
    'NORP': '#5DADE2',        # Nationalities/Religious/Political Groups
    'QUANTITY': '#EC7063',    # Light Red
    'ORDINAL': '#48C9B0',     # Turquoise
    'CARDINAL': '#F5B041',    # Amber
}

# Entity descriptions
ENTITY_DESCRIPTIONS = {
    'PERSON': 'People, including fictional characters',
    'ORG': 'Companies, agencies, institutions',
    'GPE': 'Countries, cities, states',
    'LOC': 'Non-GPE locations, mountains, water bodies',
    'DATE': 'Absolute or relative dates/periods',
    'TIME': 'Times smaller than a day',
    'MONEY': 'Monetary values, including unit',
    'PERCENT': 'Percentage values',
    'PRODUCT': 'Objects, vehicles, foods, etc.',
    'EVENT': 'Named hurricanes, battles, wars, sports events',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents made into laws',
    'LANGUAGE': 'Any named language',
    'FAC': 'Buildings, airports, highways, bridges',
    'NORP': 'Nationalities, religious or political groups',
    'QUANTITY': 'Measurements (weight, distance)',
    'ORDINAL': 'First, second, third, etc.',
    'CARDINAL': 'Numerals that do not fall under another type',
}


def load_spacy_model():
    """Load spaCy model, download if not available"""
    global nlp
    
    try:
        print(f"📦 Loading spaCy model: {SPACY_MODEL}...")
        nlp = spacy.load(SPACY_MODEL)
        print(f"✅ spaCy model loaded successfully!")
    except OSError:
        print(f"⬇️ Downloading spaCy model: {SPACY_MODEL}...")
        os.system(f"python -m spacy download {SPACY_MODEL}")
        nlp = spacy.load(SPACY_MODEL)
        print(f"✅ spaCy model downloaded and loaded!")
    
    return nlp


def load_llm_model():
    """Load LLM for context enhancement"""
    global llm_classifier
    
    if not USE_LLM_ENHANCEMENT:
        print("⏭️ LLM enhancement disabled")
        return None
    
    try:
        print("🤖 Loading LLM for context enhancement...")
        # Using a zero-shot classification model for entity context
        llm_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        print("✅ LLM loaded successfully!")
        return llm_classifier
    except Exception as e:
        print(f"⚠️ Failed to load LLM: {e}")
        print("Continuing with spaCy only...")
        return None


def extract_entities_spacy(text):
    """Extract entities using spaCy"""
    global nlp
    
    if nlp is None:
        load_spacy_model()
    
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'description': ENTITY_DESCRIPTIONS.get(ent.label_, 'Unknown entity type'),
            'color': ENTITY_COLORS.get(ent.label_, '#999999')
        })
    
    return entities, doc


def enhance_with_llm(entities, original_text):
    """Enhance entity information using LLM for context"""
    global llm_classifier
    
    if llm_classifier is None or not entities:
        return entities
    
    enhanced_entities = []
    
    for entity in entities:
        enhanced = entity.copy()
        
        try:
            # Create context around the entity
            entity_text = entity['text']
            
            # Classify the entity's role/context in the sentence
            if entity['label'] == 'PERSON':
                candidate_labels = ['politician', 'athlete', 'artist', 'scientist', 'business person', 'historical figure', 'fictional character']
            elif entity['label'] == 'ORG':
                candidate_labels = ['technology company', 'government agency', 'educational institution', 'non-profit', 'sports team', 'media company', 'financial institution']
            elif entity['label'] in ['GPE', 'LOC']:
                candidate_labels = ['capital city', 'country', 'state/province', 'tourist destination', 'historical site', 'business hub']
            elif entity['label'] == 'PRODUCT':
                candidate_labels = ['electronic device', 'vehicle', 'software', 'food/beverage', 'pharmaceutical', 'consumer product']
            elif entity['label'] == 'EVENT':
                candidate_labels = ['sports event', 'political event', 'natural disaster', 'cultural event', 'historical event', 'conference']
            else:
                # Skip LLM enhancement for other entity types
                enhanced_entities.append(enhanced)
                continue
            
            # Run classification
            result = llm_classifier(
                f"The text mentions '{entity_text}' in this context: {original_text[:500]}",
                candidate_labels,
                multi_label=False
            )
            
            # Add context information
            enhanced['context'] = result['labels'][0]
            enhanced['confidence'] = round(result['scores'][0] * 100, 1)
            
        except Exception as e:
            enhanced['context'] = 'Unable to determine'
            enhanced['confidence'] = 0
        
        enhanced_entities.append(enhanced)
    
    return enhanced_entities


def create_annotated_text(text, entities):
    """Create HTML annotated text with highlighted entities"""
    if not entities:
        return text
    
    # Sort entities by start position (reverse order for replacement)
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    annotated = text
    for ent in sorted_entities:
        start, end = ent['start'], ent['end']
        color = ent['color']
        label = ent['label']
        
        # Create highlighted span
        highlighted = f'<mark style="background-color: {color}; padding: 2px 6px; border-radius: 4px; margin: 0 2px;" data-entity="{label}" title="{ent["description"]}">{ent["text"]}<sup style="font-size: 0.7em; margin-left: 2px; font-weight: bold;">{label}</sup></mark>'
        
        annotated = annotated[:start] + highlighted + annotated[end:]
    
    return annotated


def get_entity_statistics(entities):
    """Calculate entity statistics"""
    stats = {}
    for ent in entities:
        label = ent['label']
        if label not in stats:
            stats[label] = {
                'count': 0,
                'entities': [],
                'color': ent['color'],
                'description': ent['description']
            }
        stats[label]['count'] += 1
        if ent['text'] not in stats[label]['entities']:
            stats[label]['entities'].append(ent['text'])
    
    return stats


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    """Extract named entities from text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        use_llm = data.get('use_llm', USE_LLM_ENHANCEMENT)
        
        if not text:
            return jsonify({'success': False, 'error': 'Please provide text to analyze'}), 400
        
        # Extract entities with spaCy
        entities, doc = extract_entities_spacy(text)
        
        # Enhance with LLM if enabled
        if use_llm and llm_classifier is not None:
            entities = enhance_with_llm(entities, text)
        
        # Create annotated text
        annotated_html = create_annotated_text(text, entities)
        
        # Get statistics
        stats = get_entity_statistics(entities)
        
        # Get sentences with entities
        sentences = []
        for sent in doc.sents:
            sent_ents = [e for e in entities if e['start'] >= sent.start_char and e['end'] <= sent.end_char]
            if sent_ents:
                sentences.append({
                    'text': sent.text,
                    'entities': sent_ents
                })
        
        return jsonify({
            'success': True,
            'entities': entities,
            'annotated_text': annotated_html,
            'statistics': stats,
            'sentences': sentences,
            'total_entities': len(entities),
            'unique_types': len(stats),
            'llm_enhanced': use_llm and llm_classifier is not None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/entity-types', methods=['GET'])
def entity_types():
    """Get all supported entity types"""
    types = []
    for label, description in ENTITY_DESCRIPTIONS.items():
        types.append({
            'label': label,
            'description': description,
            'color': ENTITY_COLORS.get(label, '#999999')
        })
    return jsonify({'types': types})


@app.route('/sample-texts', methods=['GET'])
def sample_texts():
    """Get sample texts for demo"""
    samples = [
        {
            'title': 'News Article',
            'text': 'Apple Inc. announced today that CEO Tim Cook will visit Paris, France next Monday to meet with President Emmanuel Macron. The tech giant plans to invest $2 billion in a new European headquarters. Microsoft and Google are also expanding their presence in the region.'
        },
        {
            'title': 'Historical Text',
            'text': 'On July 4, 1776, the United States declared independence from Great Britain. Thomas Jefferson, Benjamin Franklin, and John Adams were key figures in drafting the Declaration of Independence in Philadelphia. This historic document changed the course of world history.'
        },
        {
            'title': 'Sports News',
            'text': 'LeBron James led the Los Angeles Lakers to victory against the Golden State Warriors last night at the Staples Center. Stephen Curry scored 35 points for the Warriors. The NBA Finals will begin next week in Boston.'
        },
        {
            'title': 'Science Article',
            'text': 'NASA announced that the James Webb Space Telescope has discovered a new exoplanet 120 light-years from Earth. Dr. Sarah Chen from the Massachusetts Institute of Technology led the research team. The findings were published in Nature on December 15, 2024.'
        },
        {
            'title': 'Business Report',
            'text': 'Tesla\'s stock rose 5% on Tuesday after Elon Musk announced record quarterly earnings of $3.2 billion. The company delivered 500,000 vehicles in Q3 2024. Analysts at Goldman Sachs upgraded their price target to $350 per share.'
        }
    ]
    return jsonify({'samples': samples})


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    return jsonify({
        'spacy_loaded': nlp is not None,
        'spacy_model': SPACY_MODEL,
        'llm_loaded': llm_classifier is not None,
        'llm_enabled': USE_LLM_ENHANCEMENT
    })


# ===============================
# INITIALIZATION
# ===============================

print("🏷️ Named Entity Recognition - Day 41")
print("=" * 40)

try:
    load_spacy_model()
    load_llm_model()
except Exception as e:
    print(f"⚠️ Initialization error: {e}")
    print("Models will be loaded on first request.")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
