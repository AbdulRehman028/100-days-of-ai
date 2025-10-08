# 🌡️ Project 20 - Correlation Heatmap Generator

## 📌 Overview

This project visualizes **correlations between numerical features** in a dataset using **Seaborn’s heatmap**.  
It’s an essential tool for **EDA (Exploratory Data Analysis)** to understand feature relationships before model training.

![alt text](image.png)

## 🚀 How to Run

1. Activate your environment:
   ```
   venv\Scripts\activate


2. Navigate to the project folder:

    cd Day-20-Correlation-Heatmap

3. Install dependencies:

    pip install -r requirements.txt

4. Run:

    python correlation_heatmap.py

# 📈 Example Output
```
✅ Data Loaded Successfully:

      Month  Sales  Expenses  Profit  Customers  Marketing_Spend
0   January  10000      7000    3000        150             2000
1  February  12000      8000    4000        160             2500
...

📊 Correlation Matrix:
                 Sales  Expenses  Profit  Customers  Marketing_Spend
Sales             1.00      0.97    0.85       0.95             0.99
Expenses          0.97      1.00    0.77       0.91             0.96
Profit            0.85      0.77    1.00       0.83             0.86
Customers         0.95      0.91    0.83       1.00             0.94
Marketing_Spend   0.99      0.96    0.86       0.94             1.00
```

# 🧠 Learning Goals

- Understand correlation between numerical variables.

- Learn how to use Seaborn’s heatmap effectively.

- Improve your EDA skills before building ML models.

- Interpret positive and negative relationships visually.