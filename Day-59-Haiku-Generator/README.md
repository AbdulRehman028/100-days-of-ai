# Day 59 — Haiku Generator 🎋

AI-powered haiku poetry generator using **TinyLlama-1.1B-Chat** with structured prompt engineering, syllable validation, and a premium Japanese-inspired UI.

![alt text](hiku.png)

## Features

- **Structured Prompt Engineering** — System role + mood-matched few-shot examples + seasonal kigo hints
- **Syllable Validation** — Custom syllable counter checks 5-7-5 pattern on every generation
- **Best-of-N Selection** — Generates extra candidates and picks the highest-scoring haikus
- **8 Moods** — Serene, Melancholy, Joyful, Mysterious, Nature, Cosmic, Love, Dark
- **Seasonal Kigo** — Spring, Summer, Autumn, Winter imagery (traditional haiku element)
- **Collection System** — Auto-saves every generation to JSON, viewable in sidebar
- **Prompt Transparency** — See the exact system/user prompts and techniques used
- **Copy & Download** — One-click copy or download haiku as `.txt`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| AI Model | TinyLlama-1.1B-Chat (HuggingFace Transformers) |
| Fallback | GPT-2 Medium |
| Frontend | Tailwind CSS (CDN), vanilla JS with state management |
| Fonts | Noto Serif JP, Crimson Pro, Inter |

## Project Structure

```
Day-59-Haiku-Generator/
├── app.py                  # Flask backend + HaikuEngine
├── requirements.txt
├── README.md
├── templates/
│   └── index.html          # Tailwind-powered UI
├── static/
│   ├── css/style.css       # Custom animations & components
│   └── js/app.js           # State management & API layer
└── haiku_collection/       # Auto-saved haiku JSON files
```

## How It Works

1. **Prompt Building** — Constructs a chat-format prompt with haiku master persona, mood-matched example, and optional seasonal kigo
2. **Generation** — Runs TinyLlama with tuned temperature, repetition penalty, and top-k/top-p sampling
3. **Extraction** — Parses raw LLM output to find 3 clean haiku lines (strips artifacts, labels, metadata)
4. **Validation** — Counts syllables per line and checks the 5-7-5 pattern
5. **Scoring** — Ranks candidates by syllable accuracy, theme relevance, imagery, and penalizes repetition
6. **Selection** — Picks the top N haikus from all candidates

## Running

```bash
cd Day-59-Haiku-Generator
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

## Screenshot

Japanese-inspired dark UI with:
- Left panel: theme input, inspiration chips, mood grid, season pills, creativity slider
- Right panel: haiku cards with syllable badges, analysis panel, prompt engineering transparency, stats

## Prompt Engineering Techniques

- **Role-based system prompt** — "You are a master haiku poet"
- **Few-shot learning** — Mood-matched haiku example included in every prompt
- **Seasonal kigo hints** — Traditional seasonal imagery words
- **Output constraints** — "Output ONLY the haiku, nothing else"
- **Temperature tuning** — Adjustable 0.3–1.2 range
- **Repetition penalty** — 1.3x to prevent word reuse
- **Multi-attempt best-of-N** — Generates count+3 candidates, selects top N
