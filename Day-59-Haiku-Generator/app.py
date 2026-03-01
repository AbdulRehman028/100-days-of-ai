"""
Day 59 — Haiku Generator
=========================
Flask app that generates haikus on any theme using prompt-engineered
TinyLlama-1.1B-Chat with syllable validation and post-processing.

Haiku rules: 3 lines — 5 syllables / 7 syllables / 5 syllables
"""

import re
import time
import json
import random
import torch
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
COLLECTION_DIR = BASE_DIR / "haiku_collection"
COLLECTION_DIR.mkdir(exist_ok=True)

PRIMARY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "gpt2-medium"

# ── Moods / Styles ─────────────────────────────
MOODS = {
    "serene":     {"label": "Serene",      "icon": "🌸", "desc": "Peaceful, calm, meditative"},
    "melancholy": {"label": "Melancholy",  "icon": "🌧️", "desc": "Wistful, bittersweet, reflective"},
    "joyful":     {"label": "Joyful",      "icon": "☀️", "desc": "Bright, warm, celebratory"},
    "mysterious": {"label": "Mysterious",  "icon": "🌙", "desc": "Enigmatic, dreamlike, surreal"},
    "nature":     {"label": "Nature",      "icon": "🍃", "desc": "Forests, rivers, seasons, animals"},
    "cosmic":     {"label": "Cosmic",      "icon": "✨", "desc": "Stars, universe, infinity"},
    "love":       {"label": "Love",        "icon": "💕", "desc": "Tender, intimate, heartfelt"},
    "dark":       {"label": "Dark",        "icon": "🖤", "desc": "Haunting, shadowy, intense"},
}

# ── Seasons (traditional haiku element) ────────
SEASONS = {
    "spring": {"label": "Spring", "icon": "🌷", "kigo": ["cherry blossoms", "new leaves", "gentle rain", "birdsong", "melting snow"]},
    "summer": {"label": "Summer", "icon": "🌻", "kigo": ["cicadas", "sunlight", "warm breeze", "fireflies", "still pond"]},
    "autumn": {"label": "Autumn", "icon": "🍂", "kigo": ["falling leaves", "harvest moon", "cool wind", "migrating birds", "frost"]},
    "winter": {"label": "Winter", "icon": "❄️", "kigo": ["first snow", "bare branches", "frozen pond", "cold stars", "silence"]},
    "none":   {"label": "Any",    "icon": "🔮", "kigo": []},
}

# ── Sample Themes ──────────────────────────────
SAMPLE_THEMES = [
    "ocean waves", "mountain solitude", "rainy morning", "cherry blossoms",
    "moonlight", "autumn leaves", "first snow", "old temple",
    "empty road", "birdsong at dawn", "starry night", "cup of tea",
    "childhood memory", "distant thunder", "garden path", "fading light",
    "city at night", "flowing river", "quiet forest", "passing clouds",
    "broken mirror", "ancient stone", "sleeping cat", "candle flame",
]

# ── Few-shot haiku examples (syllable-annotated for clarity) ──
# Key insight: Annotating syllable counts in examples teaches the model
# the 5-7-5 pattern far better than just stating the rule.
HAIKU_EXAMPLES = {
    "serene": [
        "An old silent pond (5)\nA frog jumps in the water (7)\nSplash silence again (5)",
        "Still morning dewdrops (5)\nRest upon the lotus leaf (7)\nSunlight slowly wakes (5)",
    ],
    "melancholy": [
        "Faded photograph (5)\nSmiles frozen in amber light (7)\nTime moves on without (5)",
        "Empty rocking chair (5)\nStill swaying on the front porch (7)\nEchoes of laughter (5)",
    ],
    "joyful": [
        "Spring rain gathers up (5)\nDancing puddles in the lane (7)\nChildren splash and sing (5)",
        "Bright sunflower face (5)\nTurning toward the golden light (7)\nSummer joy abounds (5)",
    ],
    "mysterious": [
        "Fog hides the mountain (5)\nOnly whispers reach my ears (7)\nWhat sleeps in the mist (5)",
        "Behind the closed door (5)\nShadows whisper ancient tales (7)\nWho is listening (5)",
    ],
    "nature": [
        "Wind moves through the pines (5)\nA river carves its own path (7)\nNature finds a way (5)",
        "The crow has flown off (5)\nSwaying in the evening sun (7)\nA leafless willow (5)",
    ],
    "cosmic": [
        "Stars fall silently (5)\nInto the endless dark sea (7)\nGalaxies are born (5)",
        "Beyond Saturn's rings (5)\nSilence stretches out so far (7)\nWe are stardust still (5)",
    ],
    "love": [
        "Your hand touches mine (5)\nA thousand words left unsaid (7)\nHearts already know (5)",
        "Morning coffee shared (5)\nYour laughter fills the kitchen (7)\nThis is everything (5)",
    ],
    "dark": [
        "Shadows drink the light (5)\nWalls remember every scream (7)\nSilence is the ghost (5)",
        "Clock hands never stop (5)\nYet the room feels frozen still (7)\nDust collects on dreams (5)",
    ],
}


# ══════════════════════════════════════════════
# SYLLABLE COUNTER
# ══════════════════════════════════════════════
def count_syllables(word):
    """Estimate syllable count for an English word."""
    word = word.lower().strip()
    if not word or not re.search(r'[a-z]', word):
        return 0

    # Common exceptions
    exceptions = {
        "the": 1, "every": 3, "everything": 4, "beautiful": 3,
        "fire": 1, "hour": 1, "our": 1, "flower": 2, "power": 2,
        "quiet": 2, "poem": 2, "lion": 2, "being": 2, "real": 1,
        "heaven": 2, "ocean": 2, "area": 3, "idea": 3, "violet": 3,
        "evening": 2, "different": 3, "favorite": 3, "several": 3,
        "silence": 2, "ancient": 2, "patient": 2, "science": 2,
    }
    if word in exceptions:
        return exceptions[word]

    # Remove non-alpha
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0

    count = 0
    vowels = "aeiouy"
    prev_vowel = False

    for i, char in enumerate(word):
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjust for silent e
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    # Adjust for -ed endings
    if word.endswith("ed") and len(word) > 3 and word[-3] not in "td":
        count = max(count - 1, 1) if count > 1 else count

    return max(1, count)


def count_line_syllables(line):
    """Count syllables in a full line."""
    words = re.findall(r"[a-zA-Z']+", line)
    return sum(count_syllables(w) for w in words)


def validate_haiku(lines):
    """Check if 3 lines follow 5-7-5 pattern. Returns (is_valid, syllable_counts)."""
    if len(lines) != 3:
        return False, []
    counts = [count_line_syllables(l) for l in lines]
    is_valid = counts == [5, 7, 5]
    return is_valid, counts


# ══════════════════════════════════════════════
# HAIKU ENGINE
# ══════════════════════════════════════════════
class HaikuEngine:
    """
    Prompt-engineered haiku generator using TinyLlama-1.1B-Chat.

    Pipeline:
      1. Build structured prompt (system role + few-shot + theme/mood)
      2. Generate via LLM
      3. Extract & validate haiku (5-7-5)
      4. Multiple attempts for best quality
    """

    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.model_name = None
        self.model_type = None
        self.haikus_generated = 0
        self._load_model()

    def _load_model(self):
        """Load TinyLlama (preferred) or GPT-2 Medium fallback."""
        for name, model_class in [
            (PRIMARY_MODEL, "causal"),
            (FALLBACK_MODEL, "causal"),
        ]:
            try:
                print(f"  Trying {name}...")
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    name, local_files_only=True, torch_dtype=torch.float32)

                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                self.tokenizer = tok
                self.model_name = name
                self.model_type = "tinyllama" if "tinyllama" in name.lower() else "gpt2"

                self.pipe = pipeline(
                    "text-generation", model=mdl, tokenizer=tok,
                    max_new_tokens=45,      # Haiku = ~17 words ≈ 25-40 tokens
                    temperature=0.8, top_k=50, top_p=0.92,
                    repetition_penalty=1.4,
                    no_repeat_ngram_size=3,
                    do_sample=True, return_full_text=False,
                    pad_token_id=tok.eos_token_id,
                )
                params = sum(p.numel() for p in mdl.parameters()) / 1e6
                print(f"  ✓ Loaded {name} ({params:.0f}M params)")
                return
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

        raise RuntimeError("No model available!")

    # ── Prompt Building ────────────────────
    def _build_prompt(self, theme, mood, season):
        """
        Build a structured prompt for haiku generation.

        Key insight from Day 58: Simpler prompts work better with small models.
        Too many rules confuse them. Show, don't tell.
        """
        mood_label = MOODS.get(mood, MOODS["serene"])["label"]
        examples = HAIKU_EXAMPLES.get(mood, HAIKU_EXAMPLES["serene"])
        example = random.choice(examples)

        # Strip syllable annotations for the output but keep them in prompt
        # so the model sees the pattern
        example_annotated = example
        example_clean = re.sub(r'\s*\(\d+\)', '', example)

        # Pick a seasonal word (kigo) if season specified
        kigo_word = ""
        if season != "none" and season in SEASONS:
            kigo_word = random.choice(SEASONS[season]["kigo"])

        # Simpler system prompt — fewer rules, clearer constraint
        system_msg = (
            "You write haiku poems. A haiku has exactly 3 lines: "
            "line 1 has 5 syllables, line 2 has 7 syllables, line 3 has 5 syllables. "
            "Write ONLY the 3 lines. No title, no explanation."
        )

        # Pick a 2nd example (different from first) for reinforcement
        examples2 = [e for e in HAIKU_EXAMPLES.get(mood, HAIKU_EXAMPLES["serene"]) if e != example]
        example2_annotated = examples2[0] if examples2 else example

        if self.model_type == "tinyllama":
            # Show 2 annotated examples so model really learns 5-7-5
            kigo_hint = f" about {kigo_word} and" if kigo_word else ""
            user_msg = (
                f"Write a {mood_label.lower()} haiku{kigo_hint} about {theme}.\n\n"
                f"Example 1 (5-7-5):\n{example_annotated}\n\n"
                f"Example 2 (5-7-5):\n{example2_annotated}\n\n"
                f"Now write a NEW haiku about {theme} (3 short lines):"
            )
            # Output priming: start with a theme-relevant word to keep on-topic
            # Capitalize first word of theme for natural line start
            prime_word = theme.split()[0].capitalize()
            prompt = (
                f"<|system|>\n{system_msg}</s>\n"
                f"<|user|>\n{user_msg}</s>\n"
                f"<|assistant|>\n{prime_word}"
            )
        else:
            user_msg = (
                f"Write a {mood_label.lower()} haiku about {theme}."
            )
            prompt = (
                f"Haiku (5-7-5):\n\n"
                f"Example:\n{example_clean}\n\n"
                f"Theme: {theme}\n\n"
            )

        return prompt, system_msg, user_msg

    # ── Generation ─────────────────────────
    def generate_haiku(self, theme, mood="serene", season="none", temperature=0.8, count=3):
        """
        Generate haiku(s) with multiple attempts for best quality.
        Returns the best haiku(s) from several generation attempts.
        """
        start = time.time()

        mood = mood if mood in MOODS else "serene"
        season = season if season in SEASONS else "none"
        temperature = max(0.3, min(1.2, temperature))
        count = max(1, min(5, count))

        gen_kwargs = {"temperature": temperature, "max_new_tokens": 45}

        all_haikus = []
        attempts = count + 7  # Generate many extras for better selection

        for attempt_i in range(attempts):
            prompt, system_msg, user_msg = self._build_prompt(theme, mood, season)
            try:
                raw = self.pipe(prompt, **gen_kwargs)[0]["generated_text"]
                # Prepend the primed word (first word of theme) since return_full_text=False
                prime_word = theme.split()[0].capitalize()
                raw = prime_word + raw
                haiku = self._extract_haiku(raw)
                if haiku:
                    lines = haiku.split("\n")
                    # Try to trim lines to fit 5-7-5
                    lines = self._trim_lines(lines)
                    haiku = "\n".join(lines)
                    is_valid, syllables = validate_haiku(lines)
                    all_haikus.append({
                        "text": haiku,
                        "lines": lines,
                        "syllables": syllables,
                        "is_valid_575": is_valid,
                        "score": self._score_haiku(haiku, theme, is_valid, syllables),
                    })
                    # Early exit: if we already have enough valid 5-7-5 haikus
                    valid_count = sum(1 for h in all_haikus if h["is_valid_575"])
                    if valid_count >= count:
                        break
            except Exception:
                continue

        # Deduplicate by text (same haiku can appear twice)
        seen_texts = set()
        unique_haikus = []
        for h in all_haikus:
            if h["text"] not in seen_texts:
                seen_texts.add(h["text"])
                unique_haikus.append(h)

        # Sort by score (best first) and pick top `count`
        unique_haikus.sort(key=lambda h: h["score"], reverse=True)
        selected = unique_haikus[:count] if unique_haikus else []

        # If we got nothing at all, create a fallback
        if not selected:
            selected = [self._fallback_haiku(theme, mood)]

        elapsed = round(time.time() - start, 1)
        self.haikus_generated += len(selected)

        result = {
            "haikus": selected,
            "meta": {
                "theme": theme,
                "mood": mood,
                "mood_label": MOODS[mood]["label"],
                "mood_icon": MOODS[mood]["icon"],
                "season": season,
                "season_label": SEASONS[season]["label"],
                "count_requested": count,
                "count_generated": len(selected),
                "temperature": temperature,
                "attempts": attempts,
                "actual_attempts": attempt_i + 1,
                "model": self.model_name,
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
            "prompt_info": {
                "system_prompt": system_msg,
                "user_prompt": user_msg,
                "techniques": [
                    "Simplified system prompt (fewer rules = better compliance)",
                    "Syllable-annotated few-shot example",
                    f"Seasonal kigo imagery ({season})" if season != "none" else "No season constraint",
                    f"Temperature: {temperature}",
                    "Repetition penalty: 1.4",
                    "Multi-attempt best-of-N selection (early exit on valid 5-7-5)",
                    "Post-processing: line trimming to target syllable counts",
                    "Syllable validation (5-7-5)",
                ],
            },
        }

        self._save_to_collection(result)
        return result

    # ── Extraction ─────────────────────────
    def _extract_haiku(self, raw):
        """Extract a 3-line haiku from raw LLM output."""
        text = raw.strip()

        # Remove common prefixes the model likes to add
        text = re.sub(r'^(here\s*(is|are)|sure|okay|haiku|title|poem)[:\s]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*"', '', text)
        text = re.sub(r'"\s*$', '', text)
        # Remove syllable annotations the model might copy from examples
        text = re.sub(r'\s*\(\d+\)', '', text)

        # Try to find 3 lines that look like a haiku
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Remove lines that are clearly not haiku (too long, metadata, etc.)
        haiku_lines = []
        for line in lines:
            # Skip if it's a label/instruction
            if re.match(r'^(line\s*\d|syllable|note|theme|mood|example|write|new)', line, re.IGNORECASE):
                continue
            if len(line) > 45:  # Haiku lines are short
                continue
            if line.startswith('(') or line.startswith('['):
                continue
            # Clean up
            line = re.sub(r'\s*\(.*?\)\s*$', '', line)  # Remove trailing (5 syllables) etc.
            line = re.sub(r'^\d+\.\s*', '', line)  # Remove leading numbers "1. "
            line = re.sub(r'\s*/\s*', ' ', line)  # Remove "/" separators
            line = line.strip(' -—*•"\',:;')
            if line and len(line) > 1:
                haiku_lines.append(line)
            if len(haiku_lines) == 3:
                break

        if len(haiku_lines) >= 3:
            return '\n'.join(haiku_lines[:3])
        return None

    # ── Line Trimming ──────────────────────
    def _trim_lines(self, lines):
        """
        Post-process haiku lines to nudge them toward 5-7-5.
        Tries all contiguous word subsequences to find one matching target.
        """
        if len(lines) != 3:
            return lines

        target = [5, 7, 5]
        result = []

        for line, tgt in zip(lines, target):
            syls = count_line_syllables(line)

            if syls == tgt:
                result.append(line)
                continue

            words = line.split()

            # If over target, try all contiguous subsets (keep word order)
            if syls > tgt and len(words) >= 2:
                best_match = None
                # Try dropping words from end first (preserves beginning)
                for end in range(len(words), 1, -1):
                    for start in range(0, end - 1):
                        candidate = ' '.join(words[start:end])
                        if count_line_syllables(candidate) == tgt:
                            # Prefer longer substrings (more of the original line)
                            if best_match is None or len(candidate) > len(best_match):
                                best_match = candidate
                    if best_match:
                        break  # Found a match at this length, use it

                if best_match:
                    result.append(best_match)
                    continue

            # Can't fix it — keep original
            result.append(line)

        return result

    # ── Scoring ────────────────────────────
    def _score_haiku(self, haiku_text, theme, is_valid, syllables):
        """Score a haiku for quality ranking. Syllable accuracy dominates."""
        score = 0.0

        # Syllable accuracy (MOST important — 60% of score)
        if is_valid:
            score += 60
        else:
            target = [5, 7, 5]
            for actual, expected in zip(syllables, target):
                diff = abs(actual - expected)
                if diff == 0:
                    score += 20       # Perfect line
                elif diff == 1:
                    score += 10       # Close
                elif diff == 2:
                    score += 3        # Okayish
                # else: 0

        # Theme relevance — check if theme words appear (important!)
        theme_words = set(w for w in theme.lower().split() if len(w) > 2)
        haiku_lower = haiku_text.lower()
        theme_hits = sum(1 for w in theme_words if w in haiku_lower)
        if theme_words:
            score += (theme_hits / len(theme_words)) * 15  # Up to 15 points

        # Poetic quality heuristics
        # Has imagery (nature words, sensory language)
        imagery_words = {"wind", "rain", "sun", "moon", "star", "river", "flower", "leaf",
                         "snow", "dawn", "dusk", "shadow", "light", "stone", "wave", "cloud",
                         "bird", "tree", "garden", "silence", "whisper", "dream", "mist", "frost"}
        for w in imagery_words:
            if w in haiku_lower:
                score += 2

        # Penalize repetition
        words = haiku_lower.split()
        if len(words) != len(set(words)):
            score -= 5

        # Penalize too short or too long
        if len(haiku_text) < 20:
            score -= 10
        if len(haiku_text) > 120:
            score -= 10

        return round(score, 1)

    # ── Fallback ───────────────────────────
    def _fallback_haiku(self, theme, mood):
        """Return a decent pre-written haiku when generation fails."""
        fallbacks = {
            "serene": "Still water reflects\nThe mountains in perfect peace\nBreathing slowly now",
            "melancholy": "Faded photograph\nSmiles frozen in amber light\nTime moves on without",
            "joyful": "Morning light breaks through\nDancing on the garden stones\nA new day begins",
            "mysterious": "Behind the closed door\nShadows whisper ancient tales\nWho is listening",
            "nature": "Wind moves through the pines\nA river carves its own path\nNature finds a way",
            "cosmic": "Stars beyond counting\nEach one a distant fire\nWe are never lost",
            "love": "Your voice in the dark\nA lantern through the deep night\nI am always home",
            "dark": "Empty hallway waits\nThe clock stopped counting the hours\nNothing answers back",
        }
        text = fallbacks.get(mood, fallbacks["serene"])
        lines = text.split("\n")
        _, syllables = validate_haiku(lines)
        return {
            "text": text,
            "lines": lines,
            "syllables": syllables,
            "is_valid_575": syllables == [5, 7, 5],
            "score": 30.0,
        }

    # ── History / Collection ───────────────
    def _save_to_collection(self, result):
        """Save haikus to collection."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            theme_slug = re.sub(r'[^a-z0-9]+', '_', result["meta"]["theme"].lower())[:30]
            fp = COLLECTION_DIR / f"haiku_{theme_slug}_{ts}.json"
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_collection(self):
        """Get saved haiku collection (newest first)."""
        collection = []
        try:
            files = sorted(COLLECTION_DIR.glob("haiku_*.json"), reverse=True)
            for fp in files[:30]:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    collection.append({
                        "theme": data["meta"]["theme"],
                        "mood": data["meta"]["mood_label"],
                        "icon": data["meta"]["mood_icon"],
                        "count": data["meta"]["count_generated"],
                        "time": data["meta"]["timestamp"],
                        "haikus": [h["text"] for h in data["haikus"][:3]],
                    })
        except Exception:
            pass
        return collection

    def get_status(self):
        return {
            "model": self.model_name or "None",
            "model_type": self.model_type or "unknown",
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "haikus_generated": self.haikus_generated,
            "moods": {k: {"label": v["label"], "icon": v["icon"], "desc": v["desc"]}
                      for k, v in MOODS.items()},
            "seasons": {k: {"label": v["label"], "icon": v["icon"]}
                        for k, v in SEASONS.items()},
        }


# ══════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

print("=" * 50)
print("  🎋 Haiku Generator — Day 59")
print("=" * 50)

haiku_engine = HaikuEngine()

print(f"\n  Model: {haiku_engine.model_name}")
print(f"  Server: http://localhost:5000")
print("=" * 50)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    theme = data.get('theme', '').strip()
    mood = data.get('mood', 'serene')
    season = data.get('season', 'none')
    temperature = float(data.get('temperature', 0.8))
    count = int(data.get('count', 3))

    if not theme:
        return jsonify({"error": "Please enter a theme."}), 400
    if len(theme) > 80:
        return jsonify({"error": "Theme must be under 80 characters."}), 400

    result = haiku_engine.generate_haiku(
        theme=theme, mood=mood, season=season,
        temperature=temperature, count=count,
    )
    return jsonify(result)


@app.route('/status')
def status():
    return jsonify(haiku_engine.get_status())


@app.route('/collection')
def collection():
    return jsonify({"collection": haiku_engine.get_collection()})


@app.route('/themes')
def themes():
    return jsonify({"themes": SAMPLE_THEMES})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
