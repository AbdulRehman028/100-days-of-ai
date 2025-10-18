# 🌳 Day 25 - Decision Tree Classifier

## 📌 Overview
This project demonstrates how to build and visualize a Decision Tree Classifier using Scikit-Learn.  
It predicts whether a person buys a computer based on categorical features like age, income, student status, and credit rating.

## 🚀 How to Run
1. Activate environment:
   ```bash
   venv\Scripts\activate
2. Navigate to the project folder:

    cd Day-25-Decision-Tree

3. Install dependencies:

    pip install -r requirements.txt

4. Run the project:

    python decision_tree_classifier.py

# 🧠 Concepts Covered

- Label encoding for categorical data

- Training a DecisionTreeClassifier

- Visualizing decision trees

- Calculating accuracy & classification reports

- Understanding entropy-based splits

# 📊 Example Output

✅ Data Loaded Successfully:
     age income student credit_rating buys_computer
0  <=30   high      no          fair            no
1  <=30   high      no     excellent            no
...

🎯 Accuracy: 1.0
📄 Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         2

🧠 Prediction for [[1, 2, 1, 0]]: 1 (1=Yes, 0=No)

# 🌱 Learning Goals

Understand tree-based decision boundaries

Visualize model logic using plot_tree()

Learn categorical encoding and entropy

Build interpretable ML models