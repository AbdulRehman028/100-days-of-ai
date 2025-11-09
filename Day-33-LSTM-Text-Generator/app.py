from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

app = Flask(__name__)

# Load GPT-2 model (will download on first run, then cache locally)
print("🤖 Loading GPT-2 model...")
model_name = "gpt2"  # You can use "gpt2-medium" for better results (larger download)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
print("✅ Model loaded successfully!")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, text_type="story", max_length=300, temperature=0.8):
    """
    Generate text using GPT-2 model
    
    Args:
        prompt: Input text/seed
        text_type: Type of content (story, poem, script, quest)
        max_length: Maximum length of generated text
        temperature: Controls randomness (0.7-1.0 recommended)
    
    Returns:
        Generated text
    """
    # Add context based on text type
    type_prompts = {
        "story": f"Write a creative story:\n\n{prompt}",
        "poem": f"Write a beautiful poem:\n\n{prompt}",
        "script": f"Write a movie script:\n\n{prompt}",
        "quest": f"Create an adventure quest:\n\n{prompt}"
    }
    
    full_prompt = type_prompts.get(text_type, prompt)
    
    # Encode input
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt')
    
    # Set attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent repetition
            early_stopping=True
        )
    
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to show only generated part
    if generated_text.startswith(full_prompt):
        generated_text = generated_text[len(full_prompt):].strip()
    
    return generated_text

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text based on user input"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        text_type = data.get('type', 'story')
        temperature = float(data.get('temperature', 0.8))
        max_length = int(data.get('max_length', 300))
        
        if not prompt:
            return jsonify({
                'error': 'Please enter a prompt to generate text'
            }), 400
        
        # Validate inputs
        if len(prompt) < 3:
            return jsonify({
                'error': 'Prompt is too short. Please enter at least 3 characters.'
            }), 400
        
        if temperature < 0.3 or temperature > 2.0:
            temperature = 0.8
        
        if max_length < 50 or max_length > 500:
            max_length = 300
        
        # Generate text
        start_time = time.time()
        generated_text = generate_text(prompt, text_type, max_length, temperature)
        generation_time = time.time() - start_time
        
        result = {
            'prompt': prompt,
            'type': text_type,
            'generated_text': generated_text,
            'generation_time': round(generation_time, 2),
            'temperature': temperature,
            'max_length': max_length
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Error generating text: {str(e)}'
        }), 500

@app.route('/stats')
def stats():
    """Get model statistics"""
    return jsonify({
        'model': model_name,
        'model_size': 'Small (124M parameters)' if model_name == 'gpt2' else 'Medium (355M parameters)',
        'framework': 'HuggingFace Transformers + PyTorch',
        'status': 'Ready'
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI Text Generator Web App")
    print("="*80)
    print("📝 Capabilities: Stories, Poems, Scripts, Quests")
    print("🤖 Model: GPT-2 (Open Source)")
    print("🌐 Opening web interface at http://127.0.0.1:5000")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
