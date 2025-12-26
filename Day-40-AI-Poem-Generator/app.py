"""
Day 40: AI Poem Generator (Fine-Tune on Poetry Dataset)
Fine-tune LLM on poetry data to generate poems from themes
"""

import os
import json
import torch
from flask import Flask, render_template, request, jsonify
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    pipeline
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
MODEL_NAME = 'gpt2'
FINE_TUNED_MODEL_PATH = 'models/poem_generator'
POETRY_DATA_PATH = 'data/poems.txt'

# Global variables
model = None
tokenizer = None
generator = None


# TRAINING MODE CONFIGURATION

# Options:
#   'auto'   - Load fine-tuned model if exists, train only if missing (DEFAULT)
#   'always' - Always retrain model from scratch
#   'never'  - Never train, only use existing model (fails if not found)
TRAINING_MODE = 'auto'
# ===============================


def create_poetry_dataset():
    """Create poetry dataset for fine-tuning"""
    os.makedirs('data', exist_ok=True)
    
    # Sample poems for training (mix of styles)
    poems = '''
<|poem|> theme: love
Roses bloom in morning light,
Hearts entwined from dusk to night,
Love's sweet whisper fills the air,
Two souls dancing, none compare.
<|endpoem|>

<|poem|> theme: nature
The mountain stands in silent grace,
Wind whispers through the trees,
Rivers flow to distant seas,
Nature's beauty fills this place.
<|endpoem|>

<|poem|> theme: hope
When darkness falls upon your way,
And shadows cloud your sight,
Remember dawn will break the day,
And hope will bring the light.
<|endpoem|>

<|poem|> theme: life
Life is but a fleeting dream,
A river flowing to the sea,
Moments pass like morning gleam,
Embrace each breath, wild and free.
<|endpoem|>

<|poem|> theme: time
The clock ticks on without a care,
Seconds slip like grains of sand,
Yesterday floats in the air,
Tomorrow waits with open hand.
<|endpoem|>

<|poem|> theme: dreams
In slumber's realm I wander free,
Through valleys made of starlit haze,
Where fantasy meets memory,
And night transforms to golden days.
<|endpoem|>

<|poem|> theme: ocean
The waves crash upon the shore,
Salt and spray upon my face,
The ocean's song forevermore,
A timeless, vast, and sacred place.
<|endpoem|>

<|poem|> theme: friendship
A friend is like a gentle light,
That guides you through the storm,
In darkest hours, burning bright,
Their presence keeps you warm.
<|endpoem|>

<|poem|> theme: autumn
Golden leaves fall soft and slow,
Painting earth in amber hue,
Autumn winds begin to blow,
Bidding summer's warmth adieu.
<|endpoem|>

<|poem|> theme: night
The moon ascends her silver throne,
Stars like diamonds scattered wide,
In darkness, never quite alone,
Night embraces eventide.
<|endpoem|>

<|poem|> theme: rain
Raindrops fall like tears from sky,
Washing clean the dusty earth,
Clouds roll slowly drifting by,
Giving flowers second birth.
<|endpoem|>

<|poem|> theme: joy
Laughter bubbles, pure and bright,
Dancing through the summer air,
Joy explodes like morning light,
Spreading warmth beyond compare.
<|endpoem|>

<|poem|> theme: sadness
Tears fall silent in the night,
Heavy heart and weary soul,
Searching for a distant light,
Something true to make me whole.
<|endpoem|>

<|poem|> theme: winter
Snowflakes drift like frozen lace,
Blanketing the sleeping ground,
Winter's cold and stark embrace,
Hushes every earthly sound.
<|endpoem|>

<|poem|> theme: stars
A million suns so far away,
Burning bright through endless night,
Ancient light finds us today,
Stars tell stories with their light.
<|endpoem|>

<|poem|> theme: memory
Echoes of the days gone by,
Whisper softly in my mind,
Memories that never die,
Treasures of another time.
<|endpoem|>

<|poem|> theme: courage
Stand tall when storms arise,
Face the wind with steady heart,
Courage lights the bravest eyes,
Every end is just a start.
<|endpoem|>

<|poem|> theme: spring
Cherry blossoms paint the sky,
New beginnings all around,
Winter's cold has said goodbye,
Life awakens from the ground.
<|endpoem|>

<|poem|> theme: peace
Stillness settles on the lake,
Mirror surface, calm and clear,
Not a ripple, not a wake,
Peace has made its dwelling here.
<|endpoem|>

<|poem|> theme: sunset
Colors blaze across the west,
Orange, purple, gold, and red,
Sun descends to take its rest,
Painting beauty overhead.
<|endpoem|>

<|poem|> theme: freedom
Wings unfold and take to sky,
Soaring high above the earth,
Free from chains that used to tie,
Freedom proves our truest worth.
<|endpoem|>

<|poem|> theme: beauty
Beauty hides in simple things,
Dewdrops on a morning rose,
In the song a sparrow sings,
In the way the river flows.
<|endpoem|>

<|poem|> theme: solitude
Alone but never truly lost,
In silence find your deepest thought,
Solitude may have its cost,
But peace of mind cannot be bought.
<|endpoem|>

<|poem|> theme: mystery
Shadows dance in candlelight,
Secrets whisper through the halls,
Mystery cloaks the ancient night,
Wonder echoes off the walls.
<|endpoem|>

<|poem|> theme: faith
When the path grows dark and long,
And doubt begins to creep inside,
Faith will keep your spirit strong,
Let it be your trusted guide.
<|endpoem|>
'''
    
    with open(POETRY_DATA_PATH, 'w', encoding='utf-8') as f:
        f.write(poems)
    
    print(f"✅ Created poetry dataset at {POETRY_DATA_PATH}")
    return POETRY_DATA_PATH


class PoemDataset(Dataset):
    """Custom Dataset for poems"""
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.examples = []
        encodings = tokenizer(text, truncation=True, max_length=block_size * 100, return_tensors='pt')
        
        # Split into chunks
        input_ids = encodings['input_ids'][0]
        for i in range(0, len(input_ids) - block_size, block_size):
            chunk = input_ids[i:i + block_size]
            self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def fine_tune_model():
    """Fine-tune GPT-2 on poetry dataset"""
    global model, tokenizer, generator
    
    print("🎭 Starting Poetry Model Fine-Tuning...")
    
    # Create dataset if not exists
    if not os.path.exists(POETRY_DATA_PATH):
        create_poetry_dataset()
    
    # Load tokenizer and model
    print("📦 Loading base GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<|poem|>', '<|endpoem|>']}
    tokenizer.add_special_tokens(special_tokens)
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset using custom class
    print("📚 Loading poetry dataset...")
    train_dataset = PoemDataset(
        tokenizer=tokenizer,
        file_path=POETRY_DATA_PATH,
        block_size=128
    )
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Training setup
    os.makedirs(FINE_TUNED_MODEL_PATH, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    print("🚀 Training model (this may take a few minutes)...")
    num_epochs = 30
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    print("💾 Saving fine-tuned model...")
    model.save_pretrained(FINE_TUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
    
    # Create generator pipeline
    model.to('cpu')  # Move to CPU for inference
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    print("✅ Model fine-tuned and saved!")
    return True


def load_model():
    """Load fine-tuned model"""
    global model, tokenizer, generator
    
    model_exists = os.path.exists(os.path.join(FINE_TUNED_MODEL_PATH, 'config.json'))
    
    # Mode: Always retrain
    if TRAINING_MODE == 'always':
        print("🔄 TRAINING_MODE='always' - Retraining model from scratch...")
        fine_tune_model()
        return True
    
    # Mode: Never train
    if TRAINING_MODE == 'never':
        if not model_exists:
            raise FileNotFoundError(f"❌ TRAINING_MODE='never' but model not found at {FINE_TUNED_MODEL_PATH}")
        print("📦 TRAINING_MODE='never' - Loading existing model only...")
        tokenizer = GPT2Tokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(FINE_TUNED_MODEL_PATH)
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        print("✅ Poem generator loaded!")
        return True
    
    # Mode: Auto (default)
    if model_exists:
        print("📦 Loading fine-tuned poem generator...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            model = GPT2LMHeadModel.from_pretrained(FINE_TUNED_MODEL_PATH)
            generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
            print("✅ Poem generator loaded!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Retraining model...")
            fine_tune_model()
            return True
    else:
        print("🎭 No fine-tuned model found. Starting training...")
        fine_tune_model()
        return True


def generate_poem(theme, style='classic', length='medium', temperature=0.8):
    """Generate a poem based on theme"""
    global generator
    
    if generator is None:
        load_model()
    
    # Create prompt
    prompt = f"<|poem|> theme: {theme.lower()}\n"
    
    # Determine max length based on setting
    length_map = {
        'short': 80,
        'medium': 150,
        'long': 250
    }
    max_length = length_map.get(length, 150)
    
    # Generate
    try:
        outputs = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Clean up the output
        poem = generated_text.replace(prompt, '').strip()
        
        # Remove end token if present
        if '<|endpoem|>' in poem:
            poem = poem.split('<|endpoem|>')[0].strip()
        
        # Clean up any remaining special tokens
        poem = poem.replace('<|poem|>', '').strip()
        
        return poem
        
    except Exception as e:
        return f"Error generating poem: {str(e)}"


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate poem endpoint"""
    try:
        data = request.get_json()
        theme = data.get('theme', 'love')
        style = data.get('style', 'classic')
        length = data.get('length', 'medium')
        temperature = float(data.get('temperature', 0.8))
        
        if not theme:
            return jsonify({'success': False, 'error': 'Please provide a theme'}), 400
        
        poem = generate_poem(theme, style, length, temperature)
        
        # Count metrics
        lines = [l for l in poem.split('\n') if l.strip()]
        words = poem.split()
        
        return jsonify({
            'success': True,
            'poem': poem,
            'theme': theme,
            'metrics': {
                'lines': len(lines),
                'words': len(words),
                'characters': len(poem)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/themes', methods=['GET'])
def get_themes():
    """Get suggested themes"""
    themes = [
        'love', 'nature', 'hope', 'life', 'time', 'dreams', 
        'ocean', 'friendship', 'autumn', 'night', 'rain', 
        'joy', 'sadness', 'winter', 'stars', 'memory',
        'courage', 'spring', 'peace', 'sunset', 'freedom',
        'beauty', 'solitude', 'mystery', 'faith'
    ]
    return jsonify({'themes': themes})


@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model status"""
    model_exists = os.path.exists(os.path.join(FINE_TUNED_MODEL_PATH, 'config.json'))
    return jsonify({
        'model_loaded': model is not None,
        'model_saved': model_exists,
        'base_model': MODEL_NAME,
        'training_mode': TRAINING_MODE
    })


@app.route('/train', methods=['POST'])
def train_model():
    """Manually trigger training"""
    try:
        fine_tune_model()
        return jsonify({'success': True, 'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Load model on startup
print("🎭 AI Poem Generator - Day 40")
print("=" * 40)

try:
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    load_model()
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Model will be trained on first generation request.")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
