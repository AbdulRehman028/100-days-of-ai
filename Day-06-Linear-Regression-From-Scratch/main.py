import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# -----------------------------
# Utilities
# -----------------------------
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def add_bias(X: np.ndarray) -> np.ndarray:
    # Adds bias column of ones
    return np.c_[np.ones((X.shape[0], 1)), X]

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

# -----------------------------
# Linear Regression (from scratch)
# -----------------------------
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None  # weights including bias

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=False):
        X_b = add_bias(X)              # shape: (n_samples, n_features+1)
        n_samples, n_features = X_b.shape
        self.theta = np.zeros((n_features, 1))  # initialize weights

        y = y.reshape(-1, 1)

        for i in range(self.n_iters):
            y_pred = X_b @ self.theta
            gradients = (2 / n_samples) * (X_b.T @ (y_pred - y))
            self.theta -= self.lr * gradients

            if verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iter {i+1}/{self.n_iters} - MSE: {mse(y, y_pred):.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = add_bias(X)
        return (X_b @ self.theta).ravel()

class LinearRegressionNormalEq:
    def __init__(self):
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_b = add_bias(X)
        y = y.reshape(-1, 1)
        # Normal Equation: theta = (X^T X)^-1 X^T y
        # Use pinv for numerical stability
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = add_bias(X)
        return (X_b @ self.theta).ravel()

# -----------------------------
# Plotting (for single feature)
# -----------------------------
def maybe_plot_1d(X_train, y_train, X_test, y_test, y_pred_test, out_path="regression_plot.png"):
    # Only plot if there is exactly ONE feature
    if X_train.shape[1] != 1:
        return None
    plt.figure()
    plt.scatter(X_train[:, 0], y_train, label="Train", alpha=0.7)
    plt.scatter(X_test[:, 0], y_test, label="Test", alpha=0.7)
    # Line over test domain
    x_line = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100).reshape(-1, 1)
    # We don't have direct access to model here, so fit a small line via np.polyfit on (X_test, y_pred_test)
    coeffs = np.polyfit(X_test[:, 0], y_pred_test, 1)
    y_line = coeffs[0] * x_line[:, 0] + coeffs[1]
    plt.plot(x_line, y_line, label="Prediction line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression (1D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    return out_path

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Linear Regression from scratch (NumPy)")
    parser.add_argument("--data", type=str, default="sample_data.csv", help="Path to CSV file")
    parser.add_argument("--target", type=str, default="y", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Gradient Descent")
    parser.add_argument("--iters", type=int, default=2000, help="Iterations for Gradient Descent")
    parser.add_argument("--solver", type=str, choices=["gd", "normal"], default="gd", help="Choose solver: gd (gradient descent) or normal (normal equation)")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in CSV columns: {list(df.columns)}")

    X = df.drop(columns=[args.target]).values.astype(float)
    y = df[args.target].values.astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, seed=42)

    # Train
    if args.solver == "gd":
        model = LinearRegressionGD(lr=args.lr, n_iters=args.iters)
        model.fit(X_train, y_train, verbose=args.verbose)
    else:
        model = LinearRegressionNormalEq()
        model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    print("\n=== Evaluation on Test Set ===")
    print(f"MSE: {mse(y_test, y_pred):.6f}")
    print(f"MAE: {mae(y_test, y_pred):.6f}")
    print(f"R^2: {r2_score(y_test, y_pred):.6f}")

    # Plot if 1D
    plot_path = maybe_plot_1d(X_train, y_train, X_test, y_test, y_pred, out_path="regression_plot.png")
    if plot_path:
        print(f"Saved plot → {plot_path}")

if __name__ == "__main__":
    main()
