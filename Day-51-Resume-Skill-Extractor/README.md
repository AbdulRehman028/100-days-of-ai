# Day 51: Resume Skill Extractor 📄

Extract skills from resume text using spaCy NER and generate AI summaries with BART!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![spaCy](https://img.shields.io/badge/spaCy-NER-09a3d5.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Tailwind](https://img.shields.io/badge/Tailwind-CSS-38bdf8.svg)

## 🎯 What This Project Does

This project implements a **Resume Skill Extractor** combining:

- **spaCy NER**: Named Entity Recognition for text analysis
- **Pattern Matching**: Skill detection from curated categories
- **BART Summarizer**: AI-generated professional summaries
- **Profile Classification**: Automatic role detection

## 🏗️ Architecture

```
Resume Text
    │
    ▼
┌─────────────────────────┐
│   Pattern Matching      │  ← 7 skill categories (100+ skills)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   spaCy NER             │  ← Entity extraction (ORG, PRODUCT)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   BART Summarizer       │  ← facebook/bart-large-cnn
└─────────────────────────┘
    │
    ▼
Skills + Summary + Profile Type
```

## 🚀 Features

- 📄 **Skill Extraction** - 7 categories, 100+ recognized skills
- 🤖 **AI Summary** - Professional analysis with BART
- 🎯 **Profile Detection** - Auto-classify: Full Stack, DevOps, ML Engineer, etc.
- 🎨 **Tailwind UI** - Modern dark theme with gradient accents
- 📊 **Visual Stats** - Skill counts and category breakdown
- 📝 **Sample Resume** - One-click demo data

## 📦 Installation

1. **Navigate to the project folder:**
   ```bash
   cd Day-51-Resume-Skill-Extractor
   ```

2. **Create a virtual environment:**
   ```bash
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
   python -m spacy download en_core_web_sm
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open in browser:**
   ```
   http://localhost:5000
   ```

## 🧠 Skill Categories

| Category | Examples |
|----------|----------|
| **Programming Languages** | Python, JavaScript, Java, Go, Rust |
| **Frameworks** | React, Django, Node.js, Spring Boot |
| **Databases** | PostgreSQL, MongoDB, Redis |
| **Cloud & DevOps** | AWS, Docker, Kubernetes, Terraform |
| **AI/ML** | TensorFlow, PyTorch, LangChain |
| **Tools** | Git, GitHub, Jira, VS Code |
| **Soft Skills** | Leadership, Agile, Communication |

## 🎮 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/extract` | POST | Extract skills from text |
| `/status` | GET | Model status |
| `/sample` | GET | Get sample resume |

## 🔧 Customization

### Add Custom Skills
```python
SKILL_CATEGORIES = {
    'programming_languages': [
        'python', 'java', 'javascript',
        # Add more here
    ],
    # Add new categories
    'blockchain': [
        'solidity', 'web3', 'ethereum'
    ]
}
```

### Change Summarizer Model
```python
self.summarizer = pipeline(
    "summarization",
    model="t5-base",  # or other models
    max_length=200
)
```

## 🆚 Day 50 → Day 51 Evolution

| Day | Focus | Technology |
|-----|-------|------------|
| 50 | Text Correction | T5 Grammar Model |
| **51** | **Skill Extraction** | **spaCy + BART** |

## 🎓 Learning Outcomes

By building this project, you'll learn:

1. **spaCy NER** - Named Entity Recognition
2. **Pattern Matching** - Regex skill detection
3. **BART Summarization** - Text generation
4. **Profile Classification** - Rule-based categorization
5. **Tailwind CSS** - Modern UI framework

## 📚 Resources

- [spaCy Documentation](https://spacy.io/usage)
- [BART Model](https://huggingface.co/facebook/bart-large-cnn)
- [Tailwind CSS](https://tailwindcss.com/)
- [Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)

## 📝 License

This project is part of the 100 Days of AI challenge.
by AbdurRehman Baig with ❤️

---

**Day 51 of 100** - Building AI, one day at a time! 🚀