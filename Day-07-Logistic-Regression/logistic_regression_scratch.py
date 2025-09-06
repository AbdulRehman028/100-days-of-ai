import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load Dataset
# ------------------------
try:
    data = pd.read_csv("sample_data.csv")
    X = data[["Feature1", "Feature2"]].values
    y = data["Label"].values
    print("📂 Loaded dataset from sample_data.csv")
except FileNotFoundError:
    print("⚠️ sample_data.csv not found, generating random dataset instead.")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ------------------------
# Step 2: Sigmoid Function
# ------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ------------------------
# Step 3: Logistic Regression Class
# ------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

# ------------------------
# Step 4: Train & Evaluate
# ------------------------
model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X, y)
predictions = model.predict(X)

accuracy = np.mean(predictions == y)
print(f"✅ Accuracy: {accuracy:.2f}")

# ------------------------
# Step 5: Visualization
# ------------------------
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("Dataset Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
