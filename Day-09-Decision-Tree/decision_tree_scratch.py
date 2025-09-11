import numpy as np
import pandas as pd

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
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])  # AND logic

# ------------------------
# Step 2: Entropy
# ------------------------
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# ------------------------
# Step 3: Split Dataset
# ------------------------
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# ------------------------
# Step 4: Information Gain
# ------------------------
def information_gain(y, left_y, right_y):
    H = entropy(y)
    n = len(y)
    n_left, n_right = len(left_y), len(right_y)
    weighted_entropy = (n_left / n) * entropy(left_y) + (n_right / n) * entropy(right_y)
    return H - weighted_entropy

# ------------------------
# Step 5: Decision Tree Node
# ------------------------
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# ------------------------
# Step 6: Build Decision Tree
# ------------------------
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        best_gain = -1
        best_split = None

        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, X_left, y_left, X_right, y_right)

        if best_gain == -1:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feature_index, threshold, X_left, y_left, X_right, y_right = best_split
        left_node = self.build_tree(X_left, y_left, depth + 1)
        right_node = self.build_tree(X_right, y_right, depth + 1)
        return Node(feature_index, threshold, left_node, right_node)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(sample, self.root) for sample in X]

# ------------------------
# Step 7: Train & Evaluate
# ------------------------
tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)

accuracy = np.mean(predictions == y)
print(f"✅ Accuracy: {accuracy:.2f}")
