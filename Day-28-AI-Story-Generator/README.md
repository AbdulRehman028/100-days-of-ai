# ✍️ AI Story Generator using GPT-2

This project generates **creative stories** from a text prompt using **OpenAI’s GPT-2** model.

## 🧠 Features

- Generates human-like stories
- Uses Hugging Face Transformers
- Adjustable story length and randomness

## 🛠️ Installation

1. Clone the repository or navigate to the folder.
2. Create a virtual environment (optional but recommended):

   python -m venv venv
   venv\Scripts\activate  # (Windows)

- Install dependencies:

    pip install -r requirements.txt

## ▶️ Usage

- Run the script:

    python story_generator.py

- Enter your story prompt when asked — for example:

    Once upon a time in a distant galaxy,

The AI will generate a full story based on your input.

## ⚙️ Parameters

You can modify:

- max_length → controls story length

- temperature → controls creativity/randomness

- top_p and top_k → control sampling diversity

## 💡 Example Output

- Prompt:

    A young boy discovers a hidden portal in his backyard.

- Generated Story:

```
The portal shimmered with blue light as the boy stepped closer. He could feel the hum of another world beyond it. When he finally reached out his hand, everything changed...
```

## 🧰 Model Used

GPT-2 (Medium) — from the Hugging Face Transformers library