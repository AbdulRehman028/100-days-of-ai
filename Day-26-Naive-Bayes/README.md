# 📬 Day 26 - Naïve Bayes Classifier

## 📌 Overview
This project builds a simple **Spam Detection System** using the **Naïve Bayes Classifier**.  
The model learns word frequencies and predicts whether an email is *spam* or *ham (not spam)*.

## 🚀 How to Run

1. Activate your virtual environment:
   ```bash
   venv\Scripts\activate

2. Navigate to this folder:

    cd Day-26-Naive-Bayes

3. Install dependencies:

    pip install -r requirements.txt

4. Run the app:

    python naive_bayes_classifier.py

# 📈 Example Output
```
✅ Data Loaded Successfully:
                      text label
0        Free money now!!!  spam
1  Hi, are we still meeting today?   ham
...

🎯 Accuracy: 1.0
📄 Classification Report:
              precision    recall  f1-score   support
         ham       1.00      1.00      1.00         2
        spam       1.00      1.00      1.00         1

📧 'Claim your free prize now!' → Predicted: spam
📧 'Let's go for dinner tomorrow.' → Predicted: ham
📧 'Earn money easily from home' → Predicted: spam

```

# 🧠 Concept Recap

- Naïve Bayes assumes feature independence and uses Bayes’ Theorem:

P(Class∣Words)= P(Words∣Class)×P(Class)/P(Words)	​

- Despite its simplicity, it works exceptionally well for text classification tasks.