from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import os
import re

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# HuggingFace Router API (OpenAI-compatible endpoint)
API_URL = "https://router.huggingface.co/v1/chat/completions"
API_TOKEN = os.getenv("HF_API_TOKEN", "")

if not API_TOKEN:
    print("⚠️  WARNING: No HuggingFace API token found!")
    print("📝 Create a .env file with: HF_API_TOKEN=your_token_here")
else:
    print("✅ HuggingFace token loaded from .env file!")
    print("💡 Using HuggingFace Router API")
    print("🤖 Model: Llama 3.2 3B Instruct")
    print("😀 Emoji Prediction Ready!")

# Comprehensive emoji database organized by categories
EMOJI_DATABASE = {
    "emotions": {
        "happy": ["😊", "😃", "😄", "😁", "🤗", "😍", "🥰", "😘", "🤩", "☺️"],
        "sad": ["😢", "😭", "😔", "😞", "😿", "💔", "😥", "🥺", "😓"],
        "angry": ["😠", "😡", "🤬", "😤", "💢", "👿", "😾"],
        "love": ["❤️", "💕", "💖", "💗", "💓", "💘", "💝", "💞", "💟", "♥️"],
        "laugh": ["😂", "🤣", "😆", "😹", "🤪"],
        "excited": ["🤩", "🎉", "🎊", "🥳", "✨", "💫", "⭐"],
        "nervous": ["😰", "😨", "😱", "😬", "😓", "🥶"],
        "surprised": ["😲", "😮", "😯", "🤯", "😦", "😧"],
        "thinking": ["🤔", "🧐", "💭", "💡", "🤨"]
    },
    "gestures": {
        "thumbs": ["👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘"],
        "hands": ["👏", "🙌", "👐", "🤲", "🙏", "✋", "🤚", "🖐️", "💪"],
        "pointing": ["👆", "👇", "👈", "👉", "☝️", "👋"]
    },
    "activities": {
        "sports": ["⚽", "🏀", "🏈", "⚾", "🎾", "🏐", "🏉", "🎱", "🏓", "🏸", "🏒", "🏑", "🥊", "🥋"],
        "music": ["🎵", "🎶", "🎤", "🎧", "🎸", "🎹", "🎺", "🎷", "🥁", "🎻"],
        "art": ["🎨", "🖌️", "🖍️", "✏️", "📝", "📚", "📖", "📕"],
        "work": ["💼", "💻", "⌨️", "🖥️", "📱", "☎️", "📞", "📠", "💾"],
        "celebration": ["🎉", "🎊", "🎈", "🎁", "🎂", "🍰", "🎆", "🎇", "✨"]
    },
    "food": {
        "fruits": ["🍎", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🍑", "🍒", "🍍", "🥝", "🥥"],
        "vegetables": ["🥕", "🌽", "🥒", "🥦", "🍅", "🥑", "🌶️", "🫑", "🥬"],
        "meals": ["🍕", "🍔", "🍟", "🌭", "🥪", "🌮", "🌯", "🍱", "🍜", "🍝", "🍣", "🍤"],
        "desserts": ["🍰", "🎂", "🧁", "🍪", "🍩", "🍦", "🍨", "🍧", "🍫", "🍬", "🍭"],
        "drinks": ["☕", "🍵", "🧃", "🥤", "🧋", "🍹", "🍺", "🍻", "🥂", "🍷", "🥃"]
    },
    "animals": {
        "mammals": ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵"],
        "birds": ["🐔", "🐧", "🐦", "🐤", "🐣", "🐥", "🦆", "🦅", "🦉", "🦜", "🦚"],
        "marine": ["🐟", "🐠", "🐡", "🦈", "🐙", "🐚", "🦀", "🦞", "🦐", "🐬", "🐳", "🐋"],
        "insects": ["🐛", "🦋", "🐌", "🐞", "🐝", "🪲", "🪳", "🕷️"]
    },
    "nature": {
        "weather": ["☀️", "🌤️", "⛅", "🌥️", "☁️", "🌦️", "🌧️", "⛈️", "🌩️", "⚡", "❄️", "🌨️", "💨", "🌪️", "🌈"],
        "plants": ["🌱", "🌿", "🍀", "🌾", "🌳", "🌲", "🌴", "🌵", "🌷", "🌸", "🌹", "🌺", "🌻", "🌼"],
        "celestial": ["🌙", "⭐", "🌟", "✨", "💫", "☄️", "🌠", "🌌", "🪐"]
    },
    "travel": {
        "vehicles": ["🚗", "🚕", "🚙", "🚌", "🚎", "🏎️", "🚓", "🚑", "🚒", "🚐", "🚚", "🚛", "🚜", "🛵", "🏍️", "🚲"],
        "air": ["✈️", "🛫", "🛬", "🚁", "🛩️", "🚀", "🛸"],
        "places": ["🏠", "🏡", "🏢", "🏣", "🏤", "🏥", "🏦", "🏨", "🏩", "🏪", "🏫", "🏬", "🏭", "🏯", "🏰", "🗼", "🗽"]
    },
    "objects": {
        "time": ["⏰", "⏱️", "⏲️", "⏳", "⌛", "🕐", "🕑", "🕒"],
        "tools": ["🔨", "🔧", "🔩", "⚙️", "🗜️", "⚒️", "🛠️", "🔪"],
        "tech": ["💻", "🖥️", "⌨️", "🖱️", "🖨️", "💾", "💿", "📱", "☎️", "📞"],
        "money": ["💰", "💵", "💴", "💶", "💷", "💳", "💎", "⚖️"]
    },
    "symbols": {
        "hearts": ["❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💔", "❤️‍🔥", "❤️‍🩹"],
        "arrows": ["⬆️", "⬇️", "⬅️", "➡️", "↗️", "↘️", "↙️", "↖️", "↕️", "↔️", "🔄", "🔃"],
        "checks": ["✅", "✔️", "☑️", "❌", "❎", "⭕", "🚫", "⛔"],
        "stars": ["⭐", "🌟", "✨", "💫", "⚡", "🔥", "💥", "✴️", "🌠"]
    }
}

def predict_emojis_with_llm(text):
    """
    Use LLM to intelligently predict emojis based on text content, emotion, and context
    """
    if not API_TOKEN:
        return {"error": "API token not configured"}, 400
    
    # Build a comprehensive prompt for the LLM
    prompt = f"""Analyze this text and suggest the most relevant emojis:

Text: "{text}"

Instructions:
1. Identify the main emotion, topic, and context
2. Suggest 5-8 highly relevant emojis that capture:
   - The emotional tone
   - Key subjects or themes
   - Actions or activities mentioned
   - Overall mood and feeling
3. Order them from most to least relevant
4. Return ONLY the emojis, separated by spaces (no explanations)

Examples:
- "I love pizza and coffee!" → 😍 🍕 ☕ ❤️ 🤤
- "Feeling sad and lonely today" → 😢 😔 💔 😞 🥺
- "Just got promoted at work!" → 🎉 💼 🥳 ⭐ 👏
- "Going to the beach tomorrow!" → 🏖️ 🌊 ☀️ 😎 🏄

Now analyze the text above and return only the emojis:"""

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert emoji suggestion assistant. You understand emotions, context, and cultural meanings of emojis. You respond with ONLY relevant emojis, no other text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        emoji_text = data['choices'][0]['message']['content'].strip()
        
        # Extract only emoji characters from the response
        emojis = extract_emojis(emoji_text)
        
        # Fallback: if no emojis found, use keyword matching
        if not emojis:
            emojis = fallback_emoji_prediction(text)
        
        generation_time = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "text": text,
            "emojis": emojis[:8],  # Limit to 8 emojis
            "count": len(emojis[:8]),
            "generation_time": generation_time,
            "method": "LLM"
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Please try again."}, 504
    except requests.exceptions.RequestException as e:
        return {"error": f"API error: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

def extract_emojis(text):
    """
    Extract emoji characters from text using Unicode ranges
    """
    # Improved emoji pattern that handles multi-character emojis
    emoji_pattern = re.compile(
        "(?:"
        "[\U0001F600-\U0001F64F]|"  # emoticons
        "[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
        "[\U0001F680-\U0001F6FF]|"  # transport & map symbols
        "[\U0001F1E0-\U0001F1FF]|"  # flags
        "[\U00002702-\U000027B0]|"  # dingbats
        "[\U000024C2-\U0001F251]|"  # enclosed characters
        "[\U0001F900-\U0001F9FF]|"  # supplemental symbols
        "[\U0001FA00-\U0001FA6F]|"  # extended pictographs
        "[\U00002600-\U000026FF]"   # miscellaneous symbols
        ")(?:[\U0001F3FB-\U0001F3FF]|[\U0000FE0F\U0000200D]|[\U00002640\U00002642\U000026A7\U0001F3F3\U0001F308])*",
        flags=re.UNICODE
    )
    
    # Find all emoji matches
    emojis = emoji_pattern.findall(text)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for emoji in emojis:
        if emoji not in seen and emoji.strip():
            seen.add(emoji)
            result.append(emoji)
    
    return result[:8]  # Limit to 8 emojis

def fallback_emoji_prediction(text):
    """
    Fallback method using keyword matching when LLM fails
    """
    text_lower = text.lower()
    predicted_emojis = []
    
    # Emotion keywords
    emotion_keywords = {
        "happy": ["happy", "joy", "great", "awesome", "amazing", "wonderful", "love", "excited"],
        "sad": ["sad", "unhappy", "depressed", "down", "crying", "tears", "heartbroken"],
        "angry": ["angry", "mad", "furious", "annoyed", "frustrated", "rage"],
        "love": ["love", "adore", "romantic", "crush", "sweetheart", "valentine"],
        "laugh": ["laugh", "funny", "hilarious", "lol", "haha", "joke", "comedy"],
        "excited": ["excited", "thrilled", "pumped", "celebrate", "party", "yay"],
        "thinking": ["think", "wonder", "curious", "question", "hmm", "maybe"],
        "surprised": ["surprise", "shocked", "wow", "omg", "unexpected", "amazed"]
    }
    
    # Activity keywords
    activity_keywords = {
        "sports": ["football", "soccer", "basketball", "tennis", "game", "match", "sport"],
        "music": ["music", "song", "sing", "concert", "band", "guitar", "piano"],
        "work": ["work", "office", "job", "meeting", "project", "business", "career"],
        "celebration": ["birthday", "party", "celebrate", "anniversary", "congratulations"],
        "food": ["food", "eat", "hungry", "lunch", "dinner", "breakfast", "meal"],
        "travel": ["travel", "trip", "vacation", "beach", "hotel", "flight", "airport"]
    }
    
    # Check emotions
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            if emotion in EMOJI_DATABASE["emotions"]:
                predicted_emojis.extend(EMOJI_DATABASE["emotions"][emotion][:2])
    
    # Check activities
    for activity, keywords in activity_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            for category in EMOJI_DATABASE.values():
                if activity in category:
                    predicted_emojis.extend(category[activity][:2])
    
    # Food items
    food_items = ["pizza", "burger", "coffee", "tea", "cake", "chocolate", "ice cream"]
    for item in food_items:
        if item in text_lower:
            for food_cat in EMOJI_DATABASE["food"].values():
                predicted_emojis.extend(food_cat[:1])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_emojis = []
    for emoji in predicted_emojis:
        if emoji not in seen:
            seen.add(emoji)
            unique_emojis.append(emoji)
    
    return unique_emojis[:8] if unique_emojis else ["😊", "👍", "✨"]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emojis from text
    Expects JSON: {"text": "your text here"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Please provide text to analyze"}), 400
        
        if len(text) < 3:
            return jsonify({"error": "Text must be at least 3 characters long"}), 400
        
        if len(text) > 500:
            return jsonify({"error": "Text must be less than 500 characters"}), 400
        
        # Get emoji predictions from LLM
        result = predict_emojis_with_llm(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Return emoji statistics"""
    total_emojis = sum(len(emojis) for category in EMOJI_DATABASE.values() 
                      for emojis in category.values())
    
    return jsonify({
        "total_emojis": total_emojis,
        "categories": len(EMOJI_DATABASE),
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "status": "ready"
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("😀 Emoji Prediction Web App")
    print("="*80)
    print("🌐 Opening web interface at http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)