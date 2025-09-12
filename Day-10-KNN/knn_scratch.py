import numpy as np
import pandas as pd
from collections import Counter

# ------------------------
# Step 1: Load Dataset
# ------------------------
try:
    data = pd.read_csv("sample_data.csv")
    X = data[["Feature1", "Feature2"]].values
    y = data["Label"].values
    print("📂 Loaded dataset from sample_data.csv")
except FileNotFoundError:
    print("⚠️ sample_data.csv not found, generating toy dataset instead.")
    X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])  # 2 classes


# ------------------------
# Step 2: Euclidean Distance
# ------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# ------------------------
# Step 3: KNN Classifier
# ------------------------
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        # Compute all distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# ------------------------
# Step 4: Train & Evaluate
# ------------------------
knn = KNN(k=3)
knn.fit(X, y)
predictions = knn.predict(X)

accuracy = np.mean(predictions == y)
print(f"✅ Accuracy: {accuracy:.2f}")

print("🔮 Predictions:", predictions)
print("🎯 Actual:     ", y)
