# 🧠 Day 33 — LSTM Text Generator

### 🎯 Goal

Build a character-level text generator using an **LSTM neural network**.
It learns patterns from sample text and generates new text in a similar style.

---

## 🧩 Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy

---

## 🚀 How It Works

1. Load and preprocess a text file.
2. Convert characters to numeric sequences.
3. Train an **LSTM** to predict the next character in a sequence.
4. Generate new text based on a seed phrase.

---

## ⚙️ Usage

```bash
pip install -r requirements.txt
python text_generator.py
```

## 🧠 Example Output

<pre class="overflow-visible!" data-start="3899" data-end="4093"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"></div></div></pre>

🧠 Generated text:
once upon a time there was a brave hero who fough dragons and saved the world. the hero travell across the land and
meet wise sages who taught great treasues of all...
