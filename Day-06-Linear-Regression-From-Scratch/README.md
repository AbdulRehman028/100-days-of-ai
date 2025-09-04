# Day 06 – Linear Regression (From Scratch with NumPy)

## 📌 Objective
Build a linear regression model **without scikit-learn**, using:
- **Gradient Descent** for optimization
- (Optional) **Normal Equation** for closed-form solution  
Evaluate with MSE, MAE, and R². Plot if the dataset is 1D.

---

## 🧰 Tech Stack
- Python, NumPy, Pandas, Matplotlib

---

## 📁 Files
- `main.py` – CLI tool to train & evaluate linear regression
- `sample_data.csv` – tiny demo dataset
- `requirements.txt` – dependencies

---



Args

--data: path to CSV

--target: target column name

--solver: gd (gradient descent) or normal (normal equation)

--lr: learning rate (GD)

--iters: iterations (GD)

--test_size: test split ratio (default 0.2)

--verbose: print training progress