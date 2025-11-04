# 📧 Day 32 — Spam Classifier

### 🎯 Goal
Build a simple **Spam/Ham Classifier** using Naive Bayes and text vectorization (Bag-of-Words).

---

## 🧩 Tech Stack
- Python 🐍  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn

---

## 🚀 Steps
1. Load the dataset (`spam.csv`)
2. Preprocess text and encode labels
3. Split into training and test sets
4. Vectorize messages using `CountVectorizer`
5. Train a `MultinomialNB` classifier
6. Evaluate and visualize performance

---

## 📈 Output Example
✅ Accuracy: 0.96
📊 Classification Report:
precision recall f1-score support

       0       0.97      0.99      0.98       965
       1       0.93      0.85      0.89       150

accuracy                           0.96      1115


---

## 💬 Try It
Modify the `sample` variable in `spam_classifier.py` to test your own messages!

```python
sample = ["You won a free iPhone! Claim now!"]

(you can download full dataset from:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection)