"""
Day 60 - Startup Name/Idea Generator
------------------------------------
End-to-end Flask app using Hugging Face Transformers to generate startup ideas,
plus deterministic SVG logo generation for each result.
"""

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

PRIMARY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "gpt2-medium"

INDUSTRIES = [
    "fintech",
    "healthtech",
    "edtech",
    "e-commerce",
    "climate",
    "ai tooling",
    "cybersecurity",
    "gaming",
    "real estate",
    "creator economy",
]

TONES = ["bold", "minimal", "friendly", "luxury", "playful", "futuristic"]

LOGO_STYLES = ["monogram", "wordmark", "symbol", "badge"]

STOPWORDS = {
    "the",
    "for",
    "and",
    "with",
    "your",
    "you",
    "that",
    "from",
    "this",
    "into",
    "next",
    "future",
}


class StartupEngine:
    """Text generation + parsing + fallback + logo generation."""

    def __init__(self):
        self.pipe = None
        self.model_name = None
        self.model_type = None
        self.generated_count = 0
        self._load_model()

    def _load_model(self):
        for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                print(f"Trying model: {model_name}")
                tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                )
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                self.pipe = pipeline(
                    "text-generation",
                    model=mdl,
                    tokenizer=tok,
                    max_new_tokens=300,
                    temperature=0.75,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=tok.eos_token_id,
                )
                self.model_name = model_name
                self.model_type = "tinyllama" if "tinyllama" in model_name.lower() else "gpt2"
                print(f"Loaded model: {model_name}")
                return
            except Exception as exc:
                print(f"Failed {model_name}: {exc}")

        raise RuntimeError("No model could be loaded")

    def _slug(self, text):
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "startup"

    def _safe_sentence(self, text, max_len=120):
        cleaned = re.sub(r"\s+", " ", text.strip())
        cleaned = re.sub(r"^[\-:\"']+|[\-:\"']+$", "", cleaned)
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 1].rstrip() + "."
        return cleaned

    def _titleize_name(self, text):
        t = re.sub(r"[^A-Za-z0-9\s-]", "", text).strip()
        if not t:
            return "NovaForge"
        return " ".join(w[:1].upper() + w[1:] for w in t.split())[:36]

    def _keyword(self, industry, audience):
        source = f"{industry} {audience}".lower()
        words = re.findall(r"[a-z]+", source)
        words = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        return words[0] if words else "nova"

    def _build_prompt(self, industry, audience, tone, constraints, count):
        system = (
            "You are a startup strategist and naming expert. "
            "Return concise, practical startup ideas."
        )
        user = (
            f"Generate exactly {count} startup ideas for industry: {industry}.\n"
            f"Target audience: {audience}.\n"
            f"Brand tone: {tone}.\n"
            f"Constraints: {constraints or 'none'}.\n\n"
            "Use EXACT format:\n"
            "### IDEA 1\n"
            "NAME: <brand name, max 3 words>\n"
            "TAGLINE: <short tagline>\n"
            "PROBLEM: <one sentence>\n"
            "SOLUTION: <one sentence>\n"
            "LOGO_CONCEPT: <one sentence visual direction>\n\n"
            "Repeat for IDEA 2..N. No extra commentary."
        )

        if self.model_type == "tinyllama":
            prompt = (
                f"<|system|>\n{system}</s>\n"
                f"<|user|>\n{user}</s>\n"
                "<|assistant|>\n"
                "### IDEA 1\n"
                "NAME:"
            )
        else:
            prompt = f"{system}\n\n{user}\n\n### IDEA 1\nNAME:"

        return prompt, system, user

    def _parse_ideas(self, raw_text, count, industry, audience, tone):
        blocks = re.split(r"###\s*IDEA\s*\d+", raw_text, flags=re.IGNORECASE)
        ideas = []

        for block in blocks:
            if len(ideas) >= count:
                break
            name = self._extract_field(block, "NAME")
            tagline = self._extract_field(block, "TAGLINE")
            problem = self._extract_field(block, "PROBLEM")
            solution = self._extract_field(block, "SOLUTION")
            logo_concept = self._extract_field(block, "LOGO_CONCEPT")

            if not name or not tagline:
                continue

            item = {
                "name": self._titleize_name(name),
                "tagline": self._safe_sentence(tagline, 90),
                "problem": self._safe_sentence(problem or "Customers face friction in this market.", 140),
                "solution": self._safe_sentence(solution or "A lightweight AI product removes that friction.", 140),
                "logo_concept": self._safe_sentence(logo_concept or f"{tone} geometric identity for a {industry} brand.", 120),
            }
            ideas.append(item)

        if len(ideas) < count:
            ideas.extend(self._fallback_ideas(industry, audience, tone, count - len(ideas)))

        return ideas[:count]

    def _extract_field(self, block, field):
        match = re.search(rf"{field}:\s*(.+)", block, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip().split("\n")[0].strip()

    def _fallback_ideas(self, industry, audience, tone, needed):
        keyword = self._keyword(industry, audience)
        suffixes = ["Forge", "Pilot", "Flux", "Nest", "Loop", "Beacon", "Bridge", "Grid"]
        verbs = ["simplify", "accelerate", "automate", "optimize", "unlock", "improve"]

        out = []
        for idx in range(needed):
            suffix = suffixes[idx % len(suffixes)]
            name = f"{keyword[:1].upper()}{keyword[1:]}{suffix}"
            verb = verbs[idx % len(verbs)]
            out.append(
                {
                    "name": name,
                    "tagline": f"{tone.title()} products for modern {industry} teams.",
                    "problem": f"{audience.title()} struggle with fragmented workflows in {industry}.",
                    "solution": f"{name} uses AI to {verb} repetitive decisions and execution.",
                    "logo_concept": f"{tone.title()} {random.choice(LOGO_STYLES)} with clean geometry and {industry}-inspired motif.",
                }
            )
        return out

    def _palette(self, tone):
        palettes = {
            "bold": ("#FF4D4D", "#FFB703", "#0F172A"),
            "minimal": ("#111827", "#9CA3AF", "#F9FAFB"),
            "friendly": ("#2A9D8F", "#F4A261", "#264653"),
            "luxury": ("#B08968", "#F2E9E4", "#2B2D42"),
            "playful": ("#6D28D9", "#22D3EE", "#F59E0B"),
            "futuristic": ("#06B6D4", "#7C3AED", "#0B1020"),
        }
        return palettes.get(tone, palettes["minimal"])

    def build_logo_svg(self, name, industry, tone, style):
        primary, secondary, bg = self._palette(tone)
        initials = "".join(word[0] for word in name.split()[:2]).upper() or "S"
        shape = random.choice(["circle", "hex", "diamond", "pill"])

        if shape == "circle":
            mark = f'<circle cx="90" cy="90" r="56" fill="url(#g)" />'
        elif shape == "hex":
            mark = '<polygon points="90,28 142,59 142,121 90,152 38,121 38,59" fill="url(#g)" />'
        elif shape == "diamond":
            mark = '<polygon points="90,24 156,90 90,156 24,90" fill="url(#g)" />'
        else:
            mark = '<rect x="30" y="45" width="120" height="90" rx="30" fill="url(#g)" />'

        icon_map = {
            "fintech": "$",
            "healthtech": "+",
            "edtech": "E",
            "e-commerce": "C",
            "climate": "*",
            "ai tooling": "AI",
            "cybersecurity": "#",
            "gaming": "G",
            "real estate": "H",
            "creator economy": "CE",
        }
        motif = icon_map.get(industry.lower(), initials)

        style_hint = style.lower()
        subtitle = "Monogram" if "monogram" in style_hint else "Wordmark"

        svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="820" height="240" viewBox="0 0 820 240" role="img" aria-label="{name} logo">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{primary}"/>
      <stop offset="100%" stop-color="{secondary}"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#000" flood-opacity="0.25"/>
    </filter>
  </defs>
  <rect x="0" y="0" width="820" height="240" fill="{bg}" rx="18"/>
  <g filter="url(#shadow)">
    {mark}
  </g>
  <text x="90" y="100" fill="#FFFFFF" text-anchor="middle" font-size="34" font-family="Inter, Arial" font-weight="700">{motif}</text>
  <text x="200" y="102" fill="#FFFFFF" font-size="54" font-family="Inter, Arial" font-weight="700">{name}</text>
  <text x="200" y="142" fill="#D1D5DB" font-size="20" font-family="Inter, Arial">{subtitle} · {industry.title()}</text>
</svg>
""".strip()
        return svg

    def generate(self, industry, audience, tone, constraints, count=6, temperature=0.75, logo_style="monogram"):
        start = time.time()
        count = max(3, min(10, int(count)))
        temperature = max(0.3, min(1.2, float(temperature)))
        tone = tone if tone in TONES else "minimal"
        logo_style = logo_style if logo_style in LOGO_STYLES else "monogram"

        prompt, system_msg, user_msg = self._build_prompt(industry, audience, tone, constraints, count)

        try:
            raw = self.pipe(prompt, max_new_tokens=300, temperature=temperature)[0]["generated_text"]
        except Exception:
            raw = ""

        ideas = self._parse_ideas(raw, count, industry, audience, tone)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idea in ideas:
            svg = self.build_logo_svg(idea["name"], industry, tone, logo_style)
            slug = self._slug(idea["name"])
            file_name = f"logo_{slug}_{now}.svg"
            file_path = GENERATED_DIR / file_name
            file_path.write_text(svg, encoding="utf-8")
            idea["logo_svg"] = svg
            idea["logo_file"] = file_name
            idea["logo_data_uri"] = f"data:image/svg+xml;utf8,{quote(svg)}"

        self.generated_count += len(ideas)
        elapsed = round(time.time() - start, 1)

        return {
            "ideas": ideas,
            "meta": {
                "industry": industry,
                "audience": audience,
                "tone": tone,
                "logo_style": logo_style,
                "count": len(ideas),
                "temperature": temperature,
                "model": self.model_name,
                "time_seconds": elapsed,
                "generated_total": self.generated_count,
                "timestamp": datetime.now().isoformat(),
            },
            "prompt_info": {
                "system_prompt": system_msg,
                "user_prompt": user_msg,
                "techniques": [
                    "Role-based system prompt",
                    "Strict output schema",
                    "Deterministic parser + fallback ideas",
                    "Brand-tone color mapping",
                    "Auto SVG logo generation",
                ],
            },
        }


app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
engine = StartupEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_route():
    payload = request.json or {}
    industry = (payload.get("industry") or "ai tooling").strip().lower()
    audience = (payload.get("audience") or "startups").strip()
    tone = (payload.get("tone") or "minimal").strip().lower()
    constraints = (payload.get("constraints") or "").strip()
    logo_style = (payload.get("logo_style") or "monogram").strip().lower()
    count = int(payload.get("count") or 6)
    temperature = float(payload.get("temperature") or 0.75)

    if not audience:
        return jsonify({"error": "Audience is required."}), 400

    result = engine.generate(industry, audience, tone, constraints, count, temperature, logo_style)
    return jsonify(result)


@app.route("/status")
def status_route():
    return jsonify(
        {
            "model": engine.model_name,
            "model_type": engine.model_type,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "industries": INDUSTRIES,
            "tones": TONES,
            "logo_styles": LOGO_STYLES,
            "generated_total": engine.generated_count,
        }
    )


@app.route("/history")
def history_route():
    items = []
    for fp in sorted(GENERATED_DIR.glob("logo_*.svg"), reverse=True)[:40]:
        items.append({"file": fp.name, "size": fp.stat().st_size})
    return jsonify({"files": items})


if __name__ == "__main__":
    print("=" * 52)
    print(" Day 60 - Startup Name/Idea Generator")
    print("=" * 52)
    print(f"Model: {engine.model_name}")
    print("Server: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
