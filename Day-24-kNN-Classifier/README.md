# 🧠 Day 24 – k-Nearest Neighbors (kNN) Classifier

## 📌 Overview
This project implements the **k-Nearest Neighbors (kNN)** algorithm using **Scikit-Learn**.  
It predicts whether a student passes an exam based on *study_hours* and *sleep_hours*.

---

## 🚀 How to Run

1. Activate environment:
   ```bash
   venv\Scripts\activate
2. Navigate:

    cd Day-24-kNN-Classifier

3. Install dependencies:

    pip install -r requirements.txt

3. Run:

    python knn_classifier_app.py

# 🧠 Concepts Learned

- Intuition of k-Nearest Neighbors

- Impact of k on accuracy

- Feature scaling importance

- Model evaluation using confusion matrix and classification report

- How to predict new data points
```
📈 Example Output
✅ Model Trained Successfully with k=3
📊 Accuracy: 1.00

🧾 Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

Sample 1: Study=4, Sleep=7 → Predicted Pass=0
Sample 2: Study=8, Sleep=3 → Predicted Pass=1
Sample 3: Study=5, Sleep=5 → Predicted Pass=1
```

# 🧩 Try Yourself

- Change k to 1, 5, or 7 and observe accuracy.

- Plot decision boundaries for visual understanding.

- Add a new feature (e.g., attendance %) and compare performance.

```
💡 kNN is lazy learning — it memorizes training data and classifies based on closeness, not training weights.
```