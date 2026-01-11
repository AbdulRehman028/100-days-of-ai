# Day 50: Auto-Corrector (LLM-Powered) ✨

An intelligent text correction tool powered by T5 Transformer model. Corrects spelling, grammar, and punctuation errors with AI precision!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)

## 🎯 What This Project Does

This project implements an **Auto-Corrector** using Hugging Face Transformers:

- **T5 Model**: Fine-tuned for grammar correction
- **Spelling Fixes**: Corrects misspelled words
- **Grammar Correction**: Fixes grammatical errors
- **Punctuation**: Adds missing punctuation.
- **Real-time**: Instant corrections via web UI

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────┐
│   Text Preprocessing    │  ← Split into sentences
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   T5 Grammar Model      │  ← vennify/t5-base-grammar-correction
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Post-processing       │  ← Clean and format output
└─────────────────────────┘
    │
    ▼
Corrected Text
```

## 🚀 Features

- ✨ **LLM-Powered** - T5 Transformer for accurate corrections
- 🎨 **Modern Dark UI** - Beautiful, responsive interface
- 📝 **Side-by-side** - Compare original and corrected text
- 📊 **Change Tracking** - See how many corrections were made
- 💡 **Examples** - Try pre-built error examples
- ⌨️ **Keyboard Shortcuts** - Ctrl+Enter to correct

## 📦 Installation

1. **Navigate to the project folder:**
   ```
   cd Day-50-Auto-Corrector
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open in browser:**
   ```
   http://localhost:5000
   ```

## 🧠 How It Works

### The Model
Uses `vennify/t5-base-grammar-correction`, a T5 model fine-tuned specifically for grammar correction tasks.

### The Process
```python
# Input text with errors
input_text = "I cant beleive how grate this is working."

# Add grammar correction prefix
formatted = f"grammar: {input_text}"

# Model generates corrected version
output = "I can't believe how great this is working."
```

## 📊 Example Corrections

| Original | Corrected |
|----------|-----------|
| "I cant beleive this" | "I can't believe this" |
| "their going to the store" | "they're going to the store" |
| "me and him went" | "he and I went" |
| "The fox jump over" | "The fox jumps over" |

## 🎮 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/correct` | POST | Correct text |
| `/status` | GET | Model status |
| `/examples` | GET | Get example texts |

## 🔧 Customization

### Change the Model
```python
self.model_name = "prithivida/grammar_error_correcter_v1"
# or
self.model_name = "Grammarly/grammar-correction"
```

### Adjust Parameters
```python
self.corrector = pipeline(
    "text2text-generation",
    model=self.model,
    tokenizer=self.tokenizer,
    max_length=1024  # For longer texts
)
```

## 🆚 Day 49 → Day 50 Evolution

| Day | Focus | Technology |
|-----|-------|------------|
| 49 | Q&A Chatbot | LangChain + Zephyr-7B |
| **50** | **Text Correction** | **Transformers + T5** |

## 🎓 Learning Outcomes

By building this project, you'll learn:

1. **Hugging Face Transformers** - Loading and using models
2. **T5 Architecture** - Text-to-text generation
3. **Pipeline API** - Easy model inference
4. **Text Processing** - Sentence splitting, cleanup
5. **Modern UI** - Dark theme, responsive design

## 📚 Resources

- [Hugging Face T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [Grammar Correction Models](https://huggingface.co/models?search=grammar)
- [Transformers Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

## 📝 License

This project is part of the 100 Days of AI challenge.

mADE bY AbdurRehman Baig

---

**Day 50 of 100** - Halfway there! 🎉🚀
