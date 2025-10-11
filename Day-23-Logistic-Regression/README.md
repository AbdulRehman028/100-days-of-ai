# 🧠 Day 23 – Logistic Regression Classifier

## 📌 Overview
This project demonstrates Logistic Regression — one of the most popular algorithms for **binary classification** problems.  
It builds on previous projects (Perceptron) and introduces probabilistic decision boundaries.

## 🚀 Steps to Run

1. Activate your environment:
   ```bash
   venv\Scripts\activate
2. Navigate to the project:

    cd Day-23-Logistic-Regression


3. Install dependencies:

    pip install -r requirements.txt

4. Run:

    python logistic_regression_app.py

# 🧠 Concepts Learned

- Understanding Logistic Regression as a classification algorithm

- Using LogisticRegression() from Scikit-learn

- Interpreting confusion matrix and probabilities

- Comparing with Perceptron (Deterministic vs. Probabilistic)

- Visualizing model performance with Seaborn

# 📈 Example Output

```
📊 Accuracy: 1.00

🧾 Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

🔮 Predictions:
Sample 1: Predicted=0, Probability=[0.92 0.08]
Sample 2: Predicted=1, Probability=[0.04 0.96]
```