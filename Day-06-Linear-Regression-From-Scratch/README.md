# Day 06 – Linear Regression (From Scratch with NumPy)




## 🧰 Tech Stack
- Python, NumPy, Pandas, Matplotlib


## 📁 Files
- `main.py` – CLI tool to train & evaluate linear regression
- `sample_data.csv` – tiny demo dataset
- `requirements.txt` – dependencies


## ▶️ Usage
```bash
pip install -r requirements.txt
python main.py --data sample_data.csv --target y --solver gd --lr 0.05 --iters 2000 --verbose
# Or:
python main.py --data sample_data.csv --target y --solver normal

Args

--data: path to CSV

--target: target column name

--solver: gd (gradient descent) or normal (normal equation)

--lr: learning rate (GD)

--iters: iterations (GD)

--test_size: test split ratio (default 0.2)

--verbose: print training progress