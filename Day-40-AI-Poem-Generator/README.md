# Day 40: AI Poem Generator 🎭

Fine-tune GPT-2 on poetry data to generate beautiful poems from themes.

## 🌟 Features

- **Fine-Tuned Model**: GPT-2 fine-tuned on curated poetry dataset
- **Theme-Based Generation**: Enter any theme (love, nature, hope, etc.)
- **Adjustable Parameters**: Control length and creativity level
- **Beautiful UI**: Modern glass-morphism design with poetic aesthetics
- **25+ Pre-trained Themes**: Love, nature, hope, dreams, seasons, and more

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI**: Hugging Face Transformers, GPT-2
- **Training**: PyTorch, Custom Poetry Dataset
- **Frontend**: Tailwind CSS, Font Awesome

## 📦 Installation

```

cd Day-40-AI-Poem-Generator
pip install -r requirements.txt
python app.py

```

## 🚀 Usage

1. Run `python app.py`
2. Wait for model to fine-tune (first run only, ~5-10 minutes)
3. Open http://127.0.0.1:5000
4. Enter a theme and generate poems!

## ⚙️ Configuration

In `app.py`, you can change `TRAINING_MODE`:

- `'auto'` - Load saved model if exists, train if missing (default)
- `'always'` - Always retrain from scratch
- `'never'` - Only use existing model

## 📝 Sample Themes

- **Emotions**: love, joy, sadness, hope, peace
- **Nature**: ocean, mountains, rain, sunset, stars
- **Seasons**: spring, summer, autumn, winter
- **Abstract**: time, memory, dreams, freedom, courage

## 🎯 Model Details

- **Base Model**: GPT-2 (124M parameters)
- **Training Epochs**: 30
- **Dataset**: 25 curated poems with diverse themes
- **Special Tokens**: `<|poem|>`, `<|endpoem|>`

---
Day 40 of 100 Days of AI 🚀
