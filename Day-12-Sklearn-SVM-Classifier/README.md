# 🌸 Project 12 - SVM Classifier with Scikit-Learn

## 📌 Overview
This project implements a **Support Vector Machine (SVM)** classifier using the famous **Iris dataset**.  
It demonstrates classification, evaluation, and visualization of decision boundaries.

## 🚀 How to Run
1. Activate your environment:
   ```bash
   venv\Scripts\activate

2. Go to the project folder:

    cd Day-12-Sklearn-SVM-Classifier

3. Install dependencies:

    pip install -r requirements.txt

4. Run the script:

    python svm_classifier.py

# ✨ Example Output

✅ Accuracy: 0.95

📊 Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        15
           1       0.93      0.93      0.93        15
           2       0.93      0.93      0.93        15

📉 Confusion Matrix:
[[15  0  0]
 [ 0 14  1]
 [ 0  1 14]]


It also plots the decision boundary for the first two features of the Iris dataset.

# 🧠 Learning Goals

- Understand Support Vector Machines (SVMs).

- Use linear kernel for binary/multi-class classification.

- Evaluate with accuracy, confusion matrix, and classification report.

- Visualize decision boundaries in 2D.