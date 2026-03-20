## Day 60 - Startup Name/Idea Generator

Generate startup names and product ideas with creative LLM prompts, then auto-create SVG logos for each concept.

## Features

- LLM-powered startup brainstorming with strict structured output
- Inputs for industry, audience, constraints, tone, logo style, count, creativity
- Fallback idea engine when model output is incomplete
- SVG logo generation for every idea
- Tailwind-powered premium UI + custom visual effects
- Frontend state management via centralized JS store
- Prompt transparency panel (system prompt, user prompt, techniques)

## Tech Stack

- Python
- Flask
- Hugging Face Transformers
- Torch
- Tailwind CSS + Vanilla JS

## Run

cd Day-60-Startup-Name-Idea-Generator
pip install -r requirements.txt
python app.py

Open: `http://127.0.0.1:5000`

### API

- `GET /status` - model/device/options
- `POST /generate` - generate ideas + logos
- `GET /history` - generated logo files

### Example `POST /generate` payload

```json
{
  "industry": "healthtech",
  "audience": "small clinics",
  "constraints": "avoid buzzwords",
  "tone": "minimal",
  "logo_style": "monogram",
  "count": 6,
  "temperature": 0.75
}
```
