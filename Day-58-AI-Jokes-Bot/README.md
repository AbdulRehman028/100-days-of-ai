# Day 58 — AI Jokes Bot (Prompt Engineering)

A Flask web app that generates jokes on any topic using **prompt engineering** techniques with a locally-cached **TinyLlama-1.1B-Chat** model.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web_UI-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)

## What It Does

Enter any topic → choose a joke style → the AI crafts jokes using carefully engineered prompts.

**8 Joke Styles:**

| Style | Description |
|-------|-------------|
| 💬 One-Liner | Quick, punchy single-line jokes |
| 🔤 Pun / Wordplay | Clever double meanings and homophones |
| 👨 Dad Joke | Wholesome, groan-worthy classics |
| 🚪 Knock-Knock | The classic door-knocking format |
| 👀 Observational | Everyday life absurdities |
| 🌑 Dark Humor | Edgy comedy with a twist |
| 🔥 Roast / Sarcasm | Playful burns and witty sarcasm |
| 📖 Story Joke | Short setup-punchline narratives |

## Prompt Engineering Techniques

This project demonstrates core prompt engineering concepts:

1. **Role-Based System Prompts** — Each joke style has a unique comedian persona
2. **Few-Shot Examples** — 2 examples per style teach the model the expected format
3. **Structured Output Formatting** — Numbered jokes for reliable extraction
4. **Temperature Tuning** — User-controlled creativity (0.3 = safe → 1.2 = wild)
5. **Repetition Penalty** — Prevents the model from repeating itself
6. **Chat Template Formatting** — TinyLlama's `<|system|>/<|user|>/<|assistant|>` format

## Tech Stack

- **Model**: TinyLlama-1.1B-Chat (1.1B params, locally cached)
- **Backend**: Python, Flask, HuggingFace Transformers
- **Frontend**: HTML, Tailwind CSS, Vanilla JS
- **Focus**: Prompt Engineering — no fine-tuning, just smart prompting

## Project Structure

```
Day-58-AI-Jokes-Bot/
├── app.py                 # Flask app + JokeEngine + prompt templates
├── requirements.txt       # Dependencies
├── README.md
├── joke_history/          # Auto-saved joke sessions (JSON)
├── templates/
│   └── index.html         # UI template
└── static/
    ├── css/style.css      # Glass morphism styling
    └── js/app.js          # Client-side logic
```

## How to Run

```bash
# Install dependencies (if not already)
pip install -r requirements.txt

# Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

## Model

Uses **TinyLlama-1.1B-Chat-v1.0** (auto-detected from HuggingFace cache). Falls back to GPT-2 Medium if TinyLlama isn't available.

## Key Feature: Prompt Transparency

After generating jokes, the UI shows the **exact prompts** used — system prompt, user prompt, and all techniques applied. This makes the prompt engineering process visible and educational.
