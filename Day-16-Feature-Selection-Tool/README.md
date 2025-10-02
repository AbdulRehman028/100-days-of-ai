# 🚀 Project 16 - Feature Selection Tool

## 📌 Overview
This project delivers a command-line helper that trims your dataset down to the most informative features using **Recursive Feature Elimination (RFE)** or **Mutual Information**. Supply a CSV file, point to the target column, choose a method, and receive a reduced dataset ready for modelling.

## 🧪 Sample Datasets
- `sample_data.csv` — bite-sized demo for quick smoke tests.
- `sample_data_large.csv` — 200 synthetic customer records with extra behavioural features for more realistic experiments.

## ⚙️ Setup

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## ▶️ How to Run

```powershell
python feature_selection_tool.py --input sample_data.csv --target purchased --method mutual_info --n-features 3 --output outputs/selected_features.csv --save-scores outputs/feature_scores.csv
```

### Larger dataset example

```powershell
python feature_selection_tool.py --input sample_data_large.csv --target purchased --method rfe --n-features 6 --output outputs/selected_features_large.csv --save-scores outputs/feature_scores_large.csv
```

### Key Arguments
- `--input`: Path to your CSV dataset.
- `--target`: Name of the target column.
- `--method`: `rfe` or `mutual_info` (default: `mutual_info`).
- `--task`: Force `classification` or `regression` (otherwise inferred).
- `--n-features`: Number of features to keep (default: 5).
- `--output`: Where to store the reduced dataset (default: `selected_features.csv`).
- `--save-scores`: Optional path to persist feature rankings/scores.

## ✅ Output
- A trimmed CSV containing only the chosen features plus the target column.
- Optional CSV with feature rankings or mutual information scores.

## 🧠 Learning Goals
- Practice feature selection workflows with scikit-learn.
- Compare     RFE and mutual information techniques.
- Automate the hand-off from raw data to model-ready features.
