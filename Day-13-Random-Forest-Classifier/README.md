# 🌲 Project 13 - Random Forest Classifier with Scikit-Learn

## 📌 Overview
This project demonstrates a **Random Forest Classifier** using the Iris dataset.  
It trains the model, evaluates performance, and shows feature importance.

![alt text](image.png)

## 🚀 How to Run

1. Activate your environment:
   ```bash
   venv\Scripts\activate

2. Navigate to the project folder:

    ```bash
    cd Day-13-Random-Forest-Classifier

3. Install dependencies:

    ```pip install -r requirements.txt

4. Run the script:

    ```python random_forest_classifier.py


# ✨ Example Output

```✅ Accuracy: 0.95

📊 Classification Report:
              precision    recall  f1-score   support
setosa           1.00      1.00      1.00        15
versicolor       0.93      0.93      0.93        15
virginica        0.93      0.93      0.93        15

📉 Confusion Matrix:
[[15  0  0]
 [ 0 14  1]
 [ 0  1 14]]
The script also plots a feature importance bar chart, showing which Iris features were most useful for classification.
```

# 🧠 Learning Goals
Understand Random Forests as an ensemble of decision trees.

Train and evaluate a model using scikit-learn.

Interpret feature importances.

Compare results with SVM (Day 12) for deeper learning.