from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size 

# HuggingFace Router API
API_URL = "https://router.huggingface.co/v1/chat/completions"
API_TOKEN = os.getenv("HF_API_TOKEN", "")

if not API_TOKEN:
    print("⚠️  WARNING: No HuggingFace API token found!")
    print("📝 Create a .env file with: HF_API_TOKEN=your_token_here")
else:
    print("✅ HuggingFace token loaded!")
    print("🤖 Model: Llama 3.2 3B Instruct")
    print("😂 Meme Caption Generator Ready!")

# Meme caption styles/templates
CAPTION_STYLES = {
    "funny": "Generate a funny, humorous caption that would make people laugh",
    "relatable": "Generate a relatable caption that captures everyday situations",
    "sarcastic": "Generate a sarcastic, witty caption with clever wordplay",
    "motivational": "Generate an inspirational, motivational caption",
    "dark_humor": "Generate a dark humor caption with edgy comedy",
    "wholesome": "Generate a wholesome, heartwarming caption",
    "absurd": "Generate an absurd, surreal caption that doesn't make logical sense",
    "pop_culture": "Generate a caption referencing popular culture, memes, or trends"
}

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', styles=CAPTION_STYLES)

@app.route('/generate', methods=['POST'])
def generate_captions():
    """Generate meme captions based on image description"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        style = data.get('style', 'funny')
        count = int(data.get('count', 5))
        
        # Validation
        if not description:
            return jsonify({"error": "Please provide an image description"}), 400
        
        if len(description) < 5:
            return jsonify({"error": "Description too short. Provide more details."}), 400
        
        if count < 1 or count > 10:
            return jsonify({"error": "Caption count must be between 1 and 10"}), 400
        
        # Generate captions using LLM
        result = generate_captions_with_llm(description, style, count)
        
        if isinstance(result, tuple):  # Error case
            return jsonify(result[0]), result[1]
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": "Invalid input format"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/stats')
def stats():
    """Return API statistics"""
    return jsonify({
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "api": "HuggingFace Router",
        "status": "ready" if API_TOKEN else "no_token"
    })

def generate_captions_with_llm(description, style, count):
    """Generate meme captions using LLM"""
    if not API_TOKEN:
        return {"error": "API token not configured"}, 400
    
    style_instruction = CAPTION_STYLES.get(style, CAPTION_STYLES["funny"])
    
    # Build comprehensive prompt for meme caption generation
    prompt = f"""You are an expert meme caption writer and internet humor specialist.

Image Description: "{description}"

Task: {style_instruction}

Requirements:
1. Generate {count} unique, creative meme captions
2. Each caption should be SHORT (max 15 words)
3. Use internet slang, meme language, and cultural references when appropriate
4. Make them punchy, memorable, and shareable
5. Vary the style and approach for each caption
6. Number each caption (1., 2., 3., etc.)
7. Each caption should stand alone and be funny/engaging on its own

Caption Style: {style.replace('_', ' ').title()}

Examples of good meme captions:
- "When you say you'll go to bed early but it's 3 AM"
- "My brain: You should panic. Me: But why? Brain: You gotta"
- "Nobody: ... Absolutely nobody: ... Me at 2 AM:"
- "They said it couldn't be done. They were right."
- "Mom can we have [thing]? We have [thing] at home. [Thing] at home:"

Now generate {count} captions for the image described above:"""

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a creative meme caption writer who understands internet culture, humor, and viral content. You write short, punchy, memorable captions that make people want to share."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.9,
        "top_p": 0.95
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        caption_text = data['choices'][0]['message']['content'].strip()
        
        # Parse captions from response
        captions = parse_captions(caption_text, count)
        
        generation_time = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "description": description,
            "style": style,
            "captions": captions,
            "count": len(captions),
            "generation_time": generation_time
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Please try again."}, 504
    except requests.exceptions.RequestException as e:
        return {"error": f"API error: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

def parse_captions(text, expected_count):
    """Parse captions from LLM response"""
    captions = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (1., 2., 1), etc.)
        import re
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
        
        # Remove quotes if present
        cleaned = cleaned.strip('"\'')
        
        if cleaned and len(cleaned) > 5:  # At least 5 characters
            captions.append(cleaned)
        
        if len(captions) >= expected_count:
            break
    
    # Fallback captions if parsing failed
    if not captions:
        captions = [
            "When the meme is so good you can't even caption it",
            "This image speaks louder than words",
            "POV: You're trying to explain this to your parents",
            "Tag someone who needs to see this",
            "Why is this so accurate though?"
        ][:expected_count]
    
    return captions[:expected_count]

if __name__ == '__main__':
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
