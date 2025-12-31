# Day 45: Fake News Detector (LLM Fact-Checking)

An AI-powered fact-checking application that analyzes news claims using multiple LLM-based approaches to detect misinformation and output verdicts.

## 🌟 Features

- **Multi-Model Analysis**: Combines multiple AI models for comprehensive fact-checking
  - 🧠 **NLI Analysis**: Natural Language Inference for claim verification
  - 🔍 **Fake News Detection**: Fine-tuned RoBERTa model
  - 😊 **Sentiment Analysis**: Detect emotional bias

- **6 Verdict Categories**:
  - ✅ **True** - Factually accurate
  - 🟢 **Mostly True** - Largely accurate
  - 🟡 **Mixed** - True and false elements
  - 🟠 **Mostly False** - Largely inaccurate
  - ❌ **False** - Factually incorrect
  - ❓ **Unverifiable** - Cannot determine

- **Red Flag Detection**: Identifies suspicious patterns
  - Sensationalist language (SHOCKING, BREAKING)
  - Conspiracy terminology
  - Absolute claims (100%, guaranteed)
  - Excessive punctuation/capitalization

- **Credibility Scoring**: 0-100% score with confidence level
- **Sample Claims**: Pre-loaded examples for testing
- **Beautiful UI**: Modern, responsive interface

## 🛠️ Tech Stack

- **Backend**: Flask, Python 3.10+
- **AI Models**: Hugging Face Transformers
  - `facebook/bart-large-mnli` (NLI)
  - `hamzab/roberta-fake-news-classification`
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Frontend**: HTML5, CSS3, JavaScript

## 📦 Installation

1. **Navigate to project directory**:
   ```bash
   cd Day-45-Fake-News-Detector
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Open in browser**:
   ```
   http://localhost:5000
   ```

## 🚀 Usage

### Analyze a Claim
1. Enter a news claim, headline, or article text
2. Click "Analyze Claim"
3. View the verdict with detailed analysis

### Sample Claims
Click on any sample card to load it for testing:
- Scientific claims
- Conspiracy theories
- Political statements
- Viral misinformation
- Factual news
- Exaggerated claims

## 📊 How It Works

### Analysis Pipeline

```
Input Claim
    │
    ├─► Red Flag Detection
    │   └─► Pattern matching for suspicious language
    │
    ├─► NLI Analysis (BART-MNLI)
    │   ├─► Factual score
    │   ├─► Misleading score
    │   ├─► Opinion score
    │   └─► Sensational score
    │
    ├─► Fake News Classification (RoBERTa)
    │   └─► REAL vs FAKE prediction
    │
    ├─► Sentiment Analysis
    │   └─► Emotional bias detection
    │
    └─► Verdict Calculation
        ├─► Credibility Score (0-100%)
        ├─► Confidence Level
        └─► Final Verdict
```

### Scoring Weights
| Factor | Weight |
|--------|--------|
| NLI Analysis | 40% |
| Fake News Model | 30% |
| Red Flags | 20% |
| Known Facts | 10% |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/check` | POST | Fact-check a claim |
| `/batch-check` | POST | Check multiple claims |
| `/verdicts` | GET | Get verdict definitions |
| `/model-status` | GET | Check model status |
| `/sample-claims` | GET | Get sample claims |

### Example API Request

```python
import requests

response = requests.post('http://localhost:5000/check', json={
    'claim': 'Scientists have discovered that drinking coffee every day can extend your lifespan by up to 10 years.'
})

data = response.json()
print(f"Verdict: {data['verdict']['verdict_info']['label']}")
print(f"Credibility: {data['verdict']['credibility_score']}%")
```

## 🚩 Red Flags Detected

The system looks for these suspicious patterns:

| Pattern | Example | Weight |
|---------|---------|--------|
| Sensationalist | SHOCKING, BREAKING | 15% |
| Conspiracy | "they don't want you to know" | 20% |
| Absolute claims | 100%, guaranteed | 10% |
| Media distrust | "mainstream media lies" | 15% |
| Miracle claims | cure-all, wonder drug | 15% |
| Excessive punctuation | !!!, ??? | 10% |

## 📈 Example Output

**Input**: "URGENT: Eating bananas and drinking warm water can cure any virus within 24 hours. Doctors are keeping this secret!!!"

**Output**:
```json
{
  "verdict": "FALSE",
  "credibility_score": 18.5,
  "confidence": 85.0,
  "reasons": [
    "NLI detects sensationalized language (72%)",
    "Fake news detector: FAKE (89%)",
    "Found 4 red flag(s): Sensationalist language, Absolute claims, Miracle claims, Excessive punctuation"
  ]
}
```

## ⚠️ Disclaimer

This is an AI-based tool and may not be 100% accurate. It should be used as one of many tools for fact-checking, not as the sole source of truth. Always:

- Verify claims from multiple reliable sources
- Check original sources when cited
- Consider the credibility of the publication
- Look for expert consensus on scientific claims

## 📚 References

- [BART Paper](https://arxiv.org/abs/1910.13461)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [Natural Language Inference](https://nlp.stanford.edu/projects/snli/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 📄 License

MIT License - Part of 100 Days of AI Challenge

---

**Day 45 of 100 Days of AI** 🚀
