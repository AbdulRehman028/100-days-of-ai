from flask import Flask, render_template, request, jsonify
import requests
import time
import os

app = Flask(__name__)

# HuggingFace NEW Router API (OpenAI-compatible endpoint)
API_URL = "https://router.huggingface.co/v1/chat/completions"
API_TOKEN = os.environ.get("HF_API_TOKEN", "")

if not API_TOKEN:
    print("⚠️  WARNING: No HuggingFace API token found!")
    print("📝 Set your token: $env:HF_API_TOKEN='your_token_here'")
else:
    print("✅ HuggingFace token found!")
    print("💡 Using NEW Router API (OpenAI-compatible)")
    print("🤖 Model: Llama 3.2 3B Instruct (Meta's latest!)")
    print("🆓 Free API - No downloads!")

def generate_text(prompt, text_type="story", max_length=300, temperature=0.9):
    """
    Generate text using HuggingFace Router API (OpenAI-compatible)
    
    Args:
        prompt: Input text/seed
        text_type: Type of content (story, poem, script, quest)
        max_length: Maximum length of generated text
        temperature: Controls creativity (0.7-1.0 recommended)
    
    Returns:
        Generated text
    """
    if not API_TOKEN:
        raise Exception("HuggingFace API token required! Set: $env:HF_API_TOKEN='your_token'")
    
    # Add context based on text type
    type_instructions = {
        "story": f"Write a creative and engaging story: {prompt}",
        "poem": f"Write a beautiful and artistic poem: {prompt}",
        "script": f"Write a movie script scene: {prompt}",
        "quest": f"Create an exciting adventure quest: {prompt}",
        "social": f"Write an engaging social media post about: {prompt}",
        "blog": f"Write a comprehensive blog post about: {prompt}",
        "email": f"Write a professional email about: {prompt}",
        "article": f"Write an informative article about: {prompt}"
    }
    
    user_message = type_instructions.get(text_type, prompt)
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Cap temperature at safe levels to prevent gibberish
    # Different content types have different optimal temperatures
    if text_type in ["poem", "script", "social"]:
        safe_temperature = min(temperature, 1.0)  # Max 1.0 for creative content
    else:
        safe_temperature = min(temperature, 0.95)  # Max 0.95 for longer content
    
    # Build specific instructions based on content type and length
    if text_type == "social":
        # Social media posts are short and punchy
        complete_message = f"{user_message}\n\nCreate an engaging social media post with:\n- A catchy hook or opening\n- Key message or value\n- Call-to-action or engaging question\n- Relevant emojis and hashtags\n- Keep it concise (150-300 words max)"
    elif text_type == "email":
        # Professional email format
        complete_message = f"{user_message}\n\nWrite a professional email with:\n- Clear subject line\n- Proper greeting\n- Well-structured body paragraphs\n- Professional tone\n- Clear call-to-action\n- Professional closing"
    elif text_type == "blog":
        # Blog post structure
        if max_length >= 1500:
            complete_message = f"{user_message}\n\nWrite a comprehensive blog post (800-1200 words) with:\n- Engaging title\n- Compelling introduction\n- Well-organized sections with subheadings\n- Examples and insights\n- Actionable takeaways\n- Strong conclusion"
        else:
            complete_message = f"{user_message}\n\nWrite a blog post (400-600 words) with clear structure, engaging content, and valuable insights."
    elif text_type == "article":
        # Article format
        if max_length >= 1500:
            complete_message = f"{user_message}\n\nWrite an informative article (800-1200 words) with:\n- Attention-grabbing headline\n- Strong lead paragraph\n- Well-researched information\n- Clear structure with subheadings\n- Supporting details and examples\n- Concluding summary"
        else:
            complete_message = f"{user_message}\n\nWrite an informative article (400-600 words) with clear information and good structure."
    elif text_type in ["story", "quest"]:
        # Narrative content with "THE END"
        if max_length >= 1500:
            complete_message = f"{user_message}\n\nWrite a DETAILED and EPIC {text_type} with RICH descriptions, character development, and an intricate plot (aim for 800-1200 words). Include vivid imagery, emotional depth, engaging dialogue, and multiple scenes. Create a comprehensive narrative with proper story structure: beginning, rising action, climax, falling action, and satisfying conclusion. Make it immersive and memorable. Always end with 'THE END'."
        elif max_length >= 800:
            complete_message = f"{user_message}\n\nWrite a COMPLETE {text_type} with good detail and depth (500-700 words). Include character development, descriptive scenes, and engaging elements. Ensure proper beginning, middle, and ending. Always conclude with 'THE END'."
        else:
            complete_message = f"{user_message}\n\nWrite a COMPLETE {text_type} with beginning, middle, and proper ending. End with 'THE END'."
    else:
        # Poem, script, or other creative content
        if max_length >= 1500:
            complete_message = f"{user_message}\n\nCreate an elaborate and detailed {text_type} with rich imagery and depth."
        else:
            complete_message = f"{user_message}\n\nCreate a complete and engaging {text_type}."
    
    # Calculate max_tokens with higher limits for detailed stories
    if max_length >= 1500:
        max_tokens = min(max_length + 1500, 3000)  # Allow up to 3000 tokens for epic stories
    else:
        max_tokens = min(max_length + 500, 2000)
    
    # Build appropriate system message based on content type
    system_messages = {
        "story": "You are a masterful storyteller. You craft engaging narratives with rich descriptions, compelling characters, and intricate plots. Every story is COMPLETE with proper structure and always ends with 'THE END'.",
        "poem": "You are a gifted poet. You create beautiful, artistic poems with vivid imagery, emotional depth, and meaningful metaphors. Your poems are complete and well-structured.",
        "script": "You are an experienced screenwriter. You write engaging scripts with natural dialogue, clear scene descriptions, and proper formatting. Your scripts are complete and compelling.",
        "quest": "You are a creative game designer. You craft exciting adventure quests with clear objectives, engaging challenges, and satisfying conclusions. Always end with 'THE END'.",
        "social": "You are a social media expert. You create engaging, concise posts that capture attention, provide value, and encourage interaction. You use emojis and hashtags effectively.",
        "blog": "You are a professional blogger. You write informative, engaging blog posts with clear structure, valuable insights, and actionable takeaways. Your content is well-researched and reader-friendly.",
        "email": "You are a professional communication specialist. You write clear, concise, and effective emails with proper structure, professional tone, and clear purpose.",
        "article": "You are a skilled journalist and content writer. You create well-researched, informative articles with strong leads, clear structure, and engaging content that informs and educates readers."
    }
    
    system_content = system_messages.get(text_type, "You are an expert creative writer who produces high-quality, complete content.")
    
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",  # Using Llama 3.2 (chat-compatible)
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": complete_message
            }
        ],
        "max_tokens": max_tokens,
        "temperature": safe_temperature,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            error_msg = response.text
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        result = response.json()
        
        # Extract generated text from OpenAI-compatible response
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text.strip()
        
        raise Exception("No text generated in response")
        
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

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
        'model': 'Llama 3.2 3B Instruct',
        'model_size': 'Medium (3B parameters)',
        'framework': 'HuggingFace Router API (OpenAI-compatible)',
        'endpoint': 'router.huggingface.co/v1',
        'capabilities': 'High-quality creative content generation',
        'storage': 'Zero local storage - API-based',
        'provider': 'Meta AI',
        'status': 'Ready'
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI Text Generator Web App")
    print("="*80)
    print("📝 Capabilities: Stories, Poems, Scripts, Quests")
    print("🤖 Using NEW HuggingFace Router API")
    print("🌐 Opening web interface at http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    if not API_TOKEN:
        print("⚠️  WARNING: Set HF_API_TOKEN to use the API!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
