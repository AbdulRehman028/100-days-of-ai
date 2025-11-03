# 🎬 IMDB Sentiment Analysis (Day 31)

A deep learning project that uses an **LSTM** network to predict movie review sentiment (positive/negative)
from the **IMDB movie review dataset** using Keras.

## 🚀 Features

- Built-in IMDB dataset from Keras
- Text preprocessing and padding
- Embedding + LSTM-based sentiment classifier
- Accuracy and loss visualization

---

## 🛠️ Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run

<pre class="overflow-visible!" data-start="1283" data-end="1328"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python imdb_sentiment_analysis.py</span></span></code></div></div></pre>

## 📊 Output

* Shows model accuracy and validation accuracy per epoch
* Evaluates on test data
* Example prediction for custom text reviews
