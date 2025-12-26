# Day 41: Named Entity Recognition 🏷️

Use spaCy for NER, enhanced with LLM for context-aware entity extraction.

## 🌟 Features

- **spaCy NER**: Fast and accurate entity extraction
- **LLM Enhancement**: Context-aware classification using BART
- **18+ Entity Types**: PERSON, ORG, GPE, DATE, MONEY, and more
- **Visual Annotations**: Color-coded entity highlighting
- **Entity Statistics**: Counts, groupings, and analysis
- **Sample Texts**: Pre-loaded examples for testing

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **NER**: spaCy (en_core_web_sm)
- **LLM**: Hugging Face Transformers (BART-MNLI)
- **Frontend**: Tailwind CSS, Font Awesome

## 📦 Installation

```bash
cd Day-41-Named-Entity-Recognition
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

## 🚀 Usage

1. Run `python app.py`
2. Open http://127.0.0.1:5000
3. Enter text or use sample texts
4. Click "Extract Entities"

## 🏷️ Supported Entity Types

| Entity | Description | Color |
|--------|-------------|-------|
| PERSON | People, characters | Red |
| ORG | Companies, institutions | Teal |
| GPE | Countries, cities | Blue |
| LOC | Locations, landmarks | Green |
| DATE | Dates and periods | Yellow |
| TIME | Times | Plum |
| MONEY | Monetary values | Mint |
| PERCENT | Percentages | Gold |
| PRODUCT | Products, objects | Purple |
| EVENT | Events | Salmon |
| WORK_OF_ART | Titles | Sky Blue |

## 🤖 LLM Enhancement

When enabled, the LLM classifies entities into subcategories:
- **PERSON**: politician, athlete, scientist, etc.
- **ORG**: tech company, government agency, etc.
- **GPE/LOC**: capital city, tourist destination, etc.

## ⚙️ Configuration

In `app.py`:
- `SPACY_MODEL`: Choose model size (sm, md, lg)
- `USE_LLM_ENHANCEMENT`: Enable/disable LLM context

---
Day 41 of 100 Days of AI 🚀
