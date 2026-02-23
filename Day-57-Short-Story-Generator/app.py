from flask import Flask, render_template, request, jsonify

from config import BASE_DIR, TEMPLATE_DIR, STATIC_DIR, GENRES, TONES, LENGTHS, SAMPLE_PROMPTS
from engine import StoryEngine


# ============================================
# FLASK APP
# ============================================
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


# ============================================
# INIT ENGINE
# ============================================
print("=" * 50)
print("Short Story Generator — LangChain + TinyLlama")
print("=" * 50)

story_engine = StoryEngine()

print(f"\nStarting server at http://localhost:5000")
print("=" * 50)


# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    plot = data.get('plot', '').strip()
    genre = data.get('genre', 'fantasy')
    tone = data.get('tone', 'light')
    length = data.get('length', 'short')
    creativity = float(data.get('creativity', 0.82))

    if not plot:
        return jsonify({"error": "Please provide a plot or premise"}), 400
    if len(plot) > 500:
        return jsonify({"error": "Plot must be under 500 characters"}), 400

    result = story_engine.generate_story(
        plot=plot, genre=genre, tone=tone,
        length=length, creativity=creativity,
    )
    return jsonify(result)


@app.route('/status')
def status():
    s = story_engine.get_status()
    return jsonify({
        "engine": s,
        "genres": {k: {"label": v["label"], "icon": v["icon"], "desc": v["desc"]}
                   for k, v in GENRES.items()},
        "tones": TONES,
        "lengths": {k: {"label": v["label"], "desc": v["desc"]}
                    for k, v in LENGTHS.items()},
    })


@app.route('/history')
def history():
    return jsonify({"stories": story_engine.get_saved_stories()})


@app.route('/prompts')
def sample_prompts():
    return jsonify({"prompts": SAMPLE_PROMPTS})


# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
