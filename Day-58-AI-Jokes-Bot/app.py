"""
Day 58 — AI Jokes Bot (Prompt Engineering)
==========================================
Flask app that uses TinyLlama-1.1B-Chat to generate jokes on any topic.

Core concept  : Prompt Engineering — crafting system prompts, few-shot examples,
                and template variables to steer a local LLM toward high-quality
                humour across multiple joke styles.

Model fallback: TinyLlama-1.1B-Chat (1.1 B) → GPT-2 Medium (355 M)
"""

import os
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
HISTORY_DIR = BASE_DIR / "joke_history"
HISTORY_DIR.mkdir(exist_ok=True)

PRIMARY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "gpt2-medium"

# ── Joke Styles & Prompt Templates ──────────

JOKE_STYLES = {
    "one-liner":    {"label": "One-Liner",        "icon": "💬", "desc": "Quick, punchy single-line jokes"},
    "pun":          {"label": "Pun / Wordplay",   "icon": "🔤", "desc": "Clever wordplay and double meanings"},
    "dad-joke":     {"label": "Dad Joke",          "icon": "👨", "desc": "Wholesome, groan-worthy classics"},
    "knock-knock":  {"label": "Knock-Knock",       "icon": "🚪", "desc": "The classic door-knocking format"},
    "observational": {"label": "Observational",    "icon": "👀", "desc": "Everyday life absurdities"},
    "dark":         {"label": "Dark Humor",        "icon": "🌑", "desc": "Edgy comedy with a twist"},
    "roast":        {"label": "Roast / Sarcasm",   "icon": "🔥", "desc": "Playful burns and witty sarcasm"},
    "story":        {"label": "Story Joke",        "icon": "📖", "desc": "Short setup-punchline narratives"},
}

# ── Few-shot examples per style (Prompt Engineering core) ──

FEW_SHOT_EXAMPLES = {
    "one-liner": [
        "I like my coffee like I like my mornings — dark and impossible to get through without.",
        "Cats are tiny furry landlords — they live in your house, judge everything, and never pay rent.",
    ],
    "pun": [
        "Why don't fish play basketball? Because they're afraid of the net!",
        "Why was the equal sign so humble? Because it knew it wasn't less than or greater than anyone else!",
    ],
    "dad-joke": [
        "What did the cheese say when it looked in the mirror? Halloumi!",
        "I'm reading a book about clocks. It's about time!",
    ],
    "knock-knock": [
        "Knock knock. Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!",
        "Knock knock. Who's there? Nobel. Nobel who? Nobel, that's why I knocked!",
    ],
    "observational": [
        "Nothing makes you feel more alive than watching that wifi bar drop to zero mid-email.",
        "I love cooking. Well, I love eating. And I refuse to pay someone to do to my food what I could barely do myself.",
    ],
    "dark": [
        "My alarm clock and I have a love-hate relationship. I love sleeping, and it hates that about me.",
        "I told my doctor I broke my arm in two places. He told me to stop going to those places.",
    ],
    "roast": [
        "Your phone's battery life is like your attention span — dies halfway through anything important.",
        "Your cooking is so bad, the smoke alarm cheers you on as your personal hype man every time you enter the kitchen.",
    ],
    "story": [
        "A man walks into a library and asks for books about paranoia. The librarian whispers, \"They're right behind you!\"",
        "A teacher asked students to use \"beans\" in a sentence. One kid said \"My father grows beans.\" Then little Johnny said \"We're all human beans.\"",
    ],
}

# ── System prompts per style (the heart of prompt engineering) ──

SYSTEM_PROMPTS = {
    "one-liner": (
        "You are a world-class stand-up comedian known for razor-sharp one-liners. "
        "Write original, clever one-liner jokes that are concise (1-2 sentences max). "
        "Each joke must be self-contained, punchy, and end with a surprising twist or punchline. "
        "Never repeat the examples. Be creative and original."
    ),
    "pun": (
        "You are a pun master and wordplay genius. Write clever puns and wordplay jokes "
        "that use double meanings, homophones, or unexpected word connections. "
        "Each pun should make the reader groan AND laugh. Keep them family-friendly and original."
    ),
    "dad-joke": (
        "You are the ultimate Dad Joke champion. Write wholesome, clean, groan-worthy "
        "dad jokes that are so bad they're good. Use simple setups with obvious-yet-satisfying "
        "punchlines. The cornier the better! Keep it family-friendly."
    ),
    "knock-knock": (
        "You are a knock-knock joke specialist. Write creative knock-knock jokes "
        "following the exact format: Knock knock. / Who's there? / [Name]. / [Name] who? / [Punchline]. "
        "The punchline must cleverly twist the name into a funny phrase."
    ),
    "observational": (
        "You are a witty observational comedian like Jerry Seinfeld. Write jokes about "
        "the absurdities of everyday life related to the given topic. "
        "Use the \"Have you ever noticed...\" or \"What's the deal with...\" style. "
        "Make relatable, insightful observations that end with a humorous twist."
    ),
    "dark": (
        "You are a clever dark humor comedian. Write jokes that have an unexpected dark twist "
        "but remain tasteful — the humor comes from surprise, not cruelty. "
        "Use irony, misdirection, and subverted expectations. Keep it smart, not offensive."
    ),
    "roast": (
        "You are a roast comedian known for playful, witty burns. Write sarcastic jokes "
        "and playful roasts about the given topic. Use exaggeration, irony, and clever comparisons. "
        "Keep it fun and lighthearted — tease, don't attack."
    ),
    "story": (
        "You are a master joke storyteller. Write short joke stories (3-5 sentences) "
        "with a clear setup and a hilarious punchline at the end. "
        "Use the classic \"A man walks into...\" or situation-based format. "
        "Build anticipation and deliver a surprising, funny ending."
    ),
}

SAMPLE_TOPICS = [
    "coffee", "cats", "programming", "gym", "weather", "pizza",
    "smartphones", "dating", "school", "cooking", "dogs", "wifi",
    "monday mornings", "traffic", "dentist", "sleeping", "math",
    "shopping", "parents", "office meetings", "video games", "robots",
]


# ══════════════════════════════════════════════
# JOKE ENGINE
# ══════════════════════════════════════════════
class JokeEngine:
    """
    Prompt-engineered joke generator using TinyLlama-1.1B-Chat.

    Prompt Engineering techniques used:
      • Role-based system prompts (comedian personas per style)
      • Few-shot examples (2 per style to set pattern)
      • Structured output formatting (numbered jokes)
      • Temperature tuning (creativity control)
      • Repetition penalty + no-repeat n-gram (variety)
      • Post-processing cleanup + deduplication
    """

    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.model_name = None
        self.model_type = None  # "tinyllama" or "gpt2"
        self.jokes_generated = 0
        self._load_model()

    # ── Model Loading ──────────────────────
    def _load_model(self):
        """Try TinyLlama-1.1B-Chat → GPT-2 Medium fallback."""

        # Attempt 1: TinyLlama (best — instruction-following chat model)
        try:
            print(f"Trying {PRIMARY_MODEL}...")
            tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL, local_files_only=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                PRIMARY_MODEL, local_files_only=True, torch_dtype=torch.float32)

            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            self.tokenizer = tok
            self.model_name = PRIMARY_MODEL
            self.model_type = "tinyllama"

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=tok,
                max_new_tokens=600,
                temperature=0.9,
                top_k=50, top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tok.eos_token_id,
            )
            params = sum(p.numel() for p in mdl.parameters()) / 1e6
            print(f"  ✓ Loaded {PRIMARY_MODEL} ({params:.0f}M params)")
            return
        except Exception as e:
            print(f"  ✗ TinyLlama failed: {e}")

        # Attempt 2: GPT-2 Medium (decent text generation)
        try:
            print(f"Trying {FALLBACK_MODEL}...")
            tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL, local_files_only=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                FALLBACK_MODEL, local_files_only=True, torch_dtype=torch.float32)

            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            self.tokenizer = tok
            self.model_name = FALLBACK_MODEL
            self.model_type = "gpt2"

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=tok,
                max_new_tokens=400,
                temperature=0.9,
                top_k=50, top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tok.eos_token_id,
            )
            params = sum(p.numel() for p in mdl.parameters()) / 1e6
            print(f"  ✓ Loaded {FALLBACK_MODEL} ({params:.0f}M params)")
            return
        except Exception as e:
            print(f"  ✗ GPT-2 Medium failed: {e}")
            raise RuntimeError("No suitable model found! Check HuggingFace cache.")

    # ── Prompt Building ────────────────────
    def _build_prompt(self, topic, style, count, temperature):
        """
        Build the full prompt using prompt engineering techniques.

        For TinyLlama: Uses the chat template format:
            <|system|>\n{system_prompt}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n

        For GPT-2: Uses a structured text prompt with few-shot examples.
        """
        system_msg = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["one-liner"])
        examples = FEW_SHOT_EXAMPLES.get(style, FEW_SHOT_EXAMPLES["one-liner"])
        style_label = JOKE_STYLES[style]["label"]

        # Build numbered few-shot examples
        examples_block = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))

        if self.model_type == "tinyllama":
            # ── TinyLlama Chat Template ──
            # Key insight: Simpler prompts work better with small models.
            # Too many rules confuse them. Keep it direct.
            user_msg = (
                f"Write {count} short, funny {style_label} jokes about \"{topic}\".\n"
                f"Number them 1, 2, 3. Keep each joke to 1-2 sentences only.\n\n"
                f"Example {style_label} joke:\n{examples[0]}"
            )
            prompt = (
                f"<|system|>\n{system_msg}</s>\n"
                f"<|user|>\n{user_msg}</s>\n"
                f"<|assistant|>\nHere are {count} {style_label} jokes about \"{topic}\":\n\n1."
            )
        else:
            # ── GPT-2 Structured Prompt (few-shot only) ──
            user_msg = (
                f"Write {count} {style_label} jokes about \"{topic}\".\n"
                f"Number each joke."
            )
            prompt = (
                f"=== {style_label} Jokes about {topic} ===\n\n"
                f"Examples:\n{examples_block}\n\n"
                f"New jokes about {topic}:\n\n"
                f"1."
            )

        return prompt, system_msg, user_msg

    # ── Generation ─────────────────────────
    def generate_jokes(self, topic, style="one-liner", count=3, temperature=0.85):
        """Generate jokes using the prompt-engineered pipeline."""
        start = time.time()

        # Validate
        style = style if style in JOKE_STYLES else "one-liner"
        count = max(1, min(count, 8))
        temperature = max(0.3, min(1.2, temperature))

        # Build prompt
        prompt, system_prompt, user_prompt = self._build_prompt(
            topic, style, count, temperature
        )

        # Compute token budget: jokes are short, cap output tightly
        tokens_per_joke = 80 if style in ("story", "knock-knock") else 40
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": min(tokens_per_joke * count + 30, 500),
        }

        # Generate
        try:
            raw_output = self.pipe(prompt, **gen_kwargs)[0]["generated_text"]
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

        elapsed = round(time.time() - start, 1)

        # Post-process
        jokes = self._extract_jokes(raw_output, style, count)

        # If we got too few jokes, do a retry with higher temperature
        if len(jokes) < count and len(jokes) > 0:
            pass  # Accept what we have
        elif len(jokes) == 0:
            # Emergency: split raw output into lines as jokes
            jokes = self._fallback_extract(raw_output, count)

        self.jokes_generated += len(jokes)

        # Build result
        result = {
            "jokes": jokes,
            "meta": {
                "topic": topic,
                "style": style,
                "style_label": JOKE_STYLES[style]["label"],
                "style_icon": JOKE_STYLES[style]["icon"],
                "count_requested": count,
                "count_generated": len(jokes),
                "temperature": temperature,
                "model": self.model_name,
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
            "prompt_engineering": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "few_shot_count": len(FEW_SHOT_EXAMPLES.get(style, [])),
                "technique_used": [
                    "Role-based system prompt",
                    "Few-shot examples (2 per style)",
                    "Numbered output formatting",
                    f"Temperature: {temperature}",
                    "Repetition penalty: 1.2",
                    "No-repeat 3-gram",
                ],
            },
        }

        # Save to history
        self._save_to_history(result)

        return result

    # ── Post-Processing ────────────────────
    def _extract_jokes(self, raw, style, target_count):
        """
        Extract individual jokes from raw LLM output.
        Handles numbered lists, line-separated, and knock-knock format.
        """
        jokes = []
        text = raw.strip()

        # Try numbered format first: "1." or "1)" or just numbers
        # The assistant output is primed with "1." so the raw starts mid-joke-1
        numbered = re.split(r'\n\s*\d+[\.\)]\s*', '\n' + text)
        numbered = [j.strip() for j in numbered if j.strip()]

        if len(numbered) >= 1:
            for joke_text in numbered:
                # Take only up to the first reasonable sentence ending
                cleaned = self._truncate_joke(joke_text, style)
                if cleaned and len(cleaned) > 8:
                    jokes.append(cleaned)
                if len(jokes) >= target_count:
                    break
        else:
            # Fallback: split by double newline
            parts = text.split('\n\n')
            for part in parts:
                part = re.sub(r'^\d+[\.\)]\s*', '', part).strip()
                cleaned = self._truncate_joke(part, style)
                if cleaned and len(cleaned) > 8:
                    jokes.append(cleaned)
                if len(jokes) >= target_count:
                    break

        # Clean each joke and deduplicate
        cleaned = []
        seen = set()
        for joke in jokes[:target_count]:
            joke = self._clean_joke(joke, style)
            key = joke.lower()[:40]
            if key not in seen and len(joke) > 8:
                seen.add(key)
                cleaned.append(joke)

        return cleaned

    def _truncate_joke(self, text, style):
        """Truncate a joke to its punchline — cut off any LLM rambling."""
        if style == "knock-knock":
            return text  # Don't truncate multi-line knock-knocks
        if style == "story":
            # Allow up to 4 sentences
            sentences = re.findall(r'[^.!?]*[.!?]', text)
            if sentences:
                return ' '.join(sentences[:4]).strip()
            return text

        # For short jokes: keep first 1-2 sentences
        sentences = re.findall(r'[^.!?]*[.!?]', text)
        if sentences:
            # One-liners get 1-2 sentences, others get 2
            limit = 2 if style in ("one-liner", "pun", "dad-joke") else 2
            return ' '.join(sentences[:limit]).strip()
        # No sentence-ending found: just take first 150 chars
        return text[:150].strip()

    def _clean_joke(self, joke, style):
        """Clean a single joke text."""
        # Remove common LLM artifacts
        joke = re.sub(r'^\s*(joke|here|sure|okay|alright)[:\s]*', '', joke, flags=re.IGNORECASE)
        joke = re.sub(r'\*+', '', joke)  # Remove markdown bold/italic
        joke = re.sub(r'^[-•]\s*', '', joke)  # Remove bullet points
        joke = re.sub(r'\s+', ' ', joke).strip()

        # Remove "Topic: X" prefix if echoed back
        joke = re.sub(r'^Topic:\s*\w+[\s\n]*Joke:\s*', '', joke, flags=re.IGNORECASE)

        # For knock-knock, preserve newlines
        if style == "knock-knock":
            joke = joke.replace('. ', '.\n').replace('? ', '?\n')
            # Fix "Knock knock." formatting
            joke = re.sub(r'([Kk]nock\s+[Kk]nock)\s*\.?\s*', r'Knock knock!\n', joke)
            joke = re.sub(r"(Who'?s there\??)\s*", r"Who's there?\n", joke)

        return joke.strip()

    def _fallback_extract(self, raw, count):
        """Emergency: just split by lines and pick sensible ones."""
        lines = [l.strip() for l in raw.split('\n') if l.strip() and len(l.strip()) > 15]
        lines = [re.sub(r'^\d+[\.\)]\s*', '', l).strip() for l in lines]
        return lines[:count]

    # ── History ────────────────────────────
    def _save_to_history(self, result):
        """Save generated jokes to JSON history file."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = re.sub(r'[^a-z0-9]+', '_', result["meta"]["topic"].lower())[:30]
            filename = HISTORY_DIR / f"jokes_{topic_slug}_{ts}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Don't crash on history save failure

    def get_history(self):
        """Get saved joke history (newest first)."""
        history = []
        try:
            files = sorted(HISTORY_DIR.glob("jokes_*.json"), reverse=True)
            for fp in files[:20]:  # Last 20
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history.append({
                        "topic": data["meta"]["topic"],
                        "style": data["meta"]["style_label"],
                        "icon": data["meta"]["style_icon"],
                        "count": data["meta"]["count_generated"],
                        "time": data["meta"]["timestamp"],
                        "jokes": data["jokes"][:3],  # Preview
                    })
        except Exception:
            pass
        return history

    def get_status(self):
        """Engine status for the UI."""
        return {
            "model": self.model_name or "None",
            "model_type": self.model_type or "unknown",
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "jokes_generated": self.jokes_generated,
            "styles": JOKE_STYLES,
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
print("  🎤 AI Jokes Bot — Prompt Engineering")
print("=" * 50)

joke_engine = JokeEngine()

print(f"\n  Model: {joke_engine.model_name}")
print(f"  Server: http://localhost:5000")
print("=" * 50)


# ── Routes ─────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    topic = data.get('topic', '').strip()
    style = data.get('style', 'one-liner')
    count = int(data.get('count', 3))
    temperature = float(data.get('temperature', 0.85))

    if not topic:
        return jsonify({"error": "Please enter a topic!"}), 400
    if len(topic) > 100:
        return jsonify({"error": "Topic must be under 100 characters."}), 400

    result = joke_engine.generate_jokes(
        topic=topic,
        style=style,
        count=count,
        temperature=temperature,
    )

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)


@app.route('/status')
def status():
    return jsonify(joke_engine.get_status())


@app.route('/history')
def history():
    return jsonify({"history": joke_engine.get_history()})


@app.route('/topics')
def topics():
    return jsonify({"topics": SAMPLE_TOPICS})


# ── Run ────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
