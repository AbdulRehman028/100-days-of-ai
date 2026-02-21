# Day 57 — Short Story Generator (LangChain Chain)

A **LangChain-powered** short story generator that chains multiple AI prompts to create complete narratives from a simple plot premise.

![alt text](chrome_RpOWCh7mqA.gif)

## Features

- **Multi-Chain Pipeline**: 4-step LangChain LCEL chain (Plan → Write → Title → Polish)
- **Instruction-Following LLM**: TinyLlama-1.1B-Chat for coherent, themed fiction
- **Smart Fallback**: Auto-detects available models (TinyLlama → GPT-2 Medium → Flan-T5)
- **8 Genres**: Fantasy, Sci-Fi, Mystery, Romance, Horror, Adventure, Drama, Comedy
- **6 Tone Options**: Dark, Light, Suspenseful, Whimsical, Emotional, Action-Packed
- **3 Length Modes**: Flash Fiction, Short Story, Medium Story
- **Creativity Control**: Adjustable temperature slider
- **Genre Seed Banks**: 24 hand-crafted literary openings for GPT-2 fallback mode
- **Quality Cliff Detection**: Auto-trims stories when model drift is detected
- **Story Analysis**: Word count, vocabulary richness, sentence stats, dialogue detection
- **Story History**: Auto-saved stories with quick access
- **Inspiration Prompts**: 12 curated plot ideas to get started
- **Copy & Download**: Export stories as text files

## Tech Stack

- **Backend**: Python, Flask
- **AI Framework**: LangChain (LCEL chains)
- **Primary LLM**: TinyLlama-1.1B-Chat (instruction-following, 1.1B params)
- **Fallback LLM**: GPT-2 Medium (355M) with genre-specific narrative seeds
- **Integration**: langchain-huggingface (`HuggingFacePipeline`)
- **Frontend**: Glass morphism UI, Tailwind CSS, Plus Jakarta Sans + Playfair Display

## Chain Architecture

```
User Input (Plot, Genre, Tone, Length)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 1: PLAN (deterministic)                │
│ Pick setting + protagonist → build outline  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 2: WRITE (LangChain LCEL Chain)        │
│                                             │
│  TinyLlama mode:                            │
│    SystemPrompt + UserPrompt → LLM → Parser │
│                                             │
│  GPT-2 fallback mode:                       │
│    NarrativeSeed → LLM → Parser → Trim     │
│                                             │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 3: TITLE (LangChain LCEL Chain)        │
│ Excerpt → LLM → TitleExtract               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 4: POLISH (deterministic)              │
│ Clean → QualityTrim → Format → Analyze      │
└─────────────────────────────────────────────┘
```

## Model Fallback Chain

| Priority | Model | Type | Params | Quality |
|----------|-------|------|--------|---------|
| 1 | TinyLlama-1.1B-Chat | Instruction-following | 1.1B | Best — follows genre/tone/plot |
| 2 | GPT-2 Medium | Text completion | 355M | Good seeds, variable continuation |
| 3 | Flan-T5-Base | Seq2Seq | 248M | Basic — short outputs |

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

TinyLlama (~2.2 GB) will be downloaded automatically on first run. If not available, the app falls back to GPT-2 Medium.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/generate` | POST | Generate a story |
| `/status` | GET | Engine status & settings |
| `/history` | GET | Previously generated stories |
| `/prompts` | GET | Sample plot prompts |
