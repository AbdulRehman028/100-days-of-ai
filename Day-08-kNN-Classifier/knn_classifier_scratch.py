import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    print("⚠️ sample_data.csv not found, generating random dataset instead.")
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ------------------------
# Step 2: Distance Function
# ------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# ------------------------
# Step 3: kNN Classifier
# ------------------------
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# ------------------------
# Step 4: Train & Test
# ------------------------
knn = KNN(k=3)
knn.fit(X, y)
predictions = knn.predict(X)

accuracy = np.mean(predictions == y)
print(f"✅ Accuracy: {accuracy:.2f}")

# ------------------------
# Step 5: Visualization
# ------------------------
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=50)
plt.title("kNN Dataset Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
