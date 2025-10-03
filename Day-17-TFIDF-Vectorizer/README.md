# 📚 Project 17 - TF-IDF Vectorizer (From Scratch)

## 📌 Overview
This project implements **TF-IDF (Term Frequency – Inverse Document Frequency)** from scratch without using Scikit-Learn.  
TF-IDF helps identify important words in documents for tasks like **text classification, information retrieval, and NLP**.


## 🚀 How to Run
1. Activate environment:
   ```bash
   venv\Scripts\activate

2. Navigate:

    cd Day-17-TFIDF-From-Scratch

3. Install requirements:

    ```pip install -r requirements.txt

4. Run:

    python tfidf_from_scratch.py

✨ Example Output
Tokenized Docs: [['machine', 'learning', 'is', 'amazing'], ...]

Vocabulary: ['ai', 'amazing', 'and', 'are', 'deep', 'drives', ...]

TF Scores (First Doc): {'ai': 0.0, 'amazing': 0.25, 'and': 0.0, ...}

IDF Scores: {'ai': 1.29, 'amazing': 1.69, 'deep': 1.29, ...}

TF-IDF Matrix:
       ai  amazing   and   are  deep  drives  fields  ...
0  0.0000   0.4225  0.00  0.00  0.00   0.000   0.000
1  0.0000   0.0000  0.00  0.00  0.322  0.422   0.000
2  0.3224   0.0000  0.25  0.00  0.00   0.000   0.322
3  0.0000   0.0000  0.00  0.25  0.322  0.000   0.000

🧠 Learning Goals

Implement TF, IDF, TF-IDF step by step.

Understand how word importance is calculated.