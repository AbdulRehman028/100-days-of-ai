# 🤖 Day 22 – Perceptron Using Scikit-Learn

## 📌 Overview
In this project, you’ll learn how to implement a **Perceptron classifier** using **Scikit-Learn** — a high-level library that automates model training, scaling, and evaluation.  
This builds directly on Day 21, where you implemented a perceptron manually.

---

## 🚀 Steps to Run

1. Activate your environment:
   ```bash
   venv\Scripts\activate

Navigate to this folder:

cd Day-22-Perceptron-Using-Sklearn


Install dependencies:

pip install -r requirements.txt


Run:

python perceptron_sklearn.py

🧠 Concepts Learned

How to use sklearn.linear_model.Perceptron

Feature scaling using StandardScaler

Model evaluation using accuracy and classification report

Comparing manual vs. library implementations

🧩 Challenge Ideas

Try adjusting eta0 (learning rate) and max_iter

Add more data points and see how it affects accuracy

Plot the decision boundary using matplotlib

Compare Perceptron with other classifiers like LogisticRegression

📈 Example Output
✅ Model Trained Successfully!

📊 Accuracy: 1.0

🧾 Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         2

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3

🔮 Predictions:
Study=4, Sleep=6 → 0
Study=8, Sleep=2 → 1