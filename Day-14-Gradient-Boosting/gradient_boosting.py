import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Try XGBoost, fallback to LightGBM if not available
try:
    from xgboost import XGBClassifier
    use_xgb = True
except ImportError:
    from lightgbm import LGBMClassifier
    use_xgb = False

# ------------------------
# Step 1: Load Dataset (Iris)
# ------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# ------------------------
# Step 2: Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------
# Step 3: Standardize Features
# ------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------
# Step 4: Train Gradient Boosting Classifier
# ------------------------
if use_xgb:
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
    print("✅ Using XGBoost Classifier")
else:
    model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    print("✅ Using LightGBM Classifier")

model.fit(X_train, y_train)

# ------------------------
# Step 5: Predictions & Evaluation
# ------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------
# Step 6: Feature Importance Visualization
# ------------------------
importances = model.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)

plt.figure(figsize=(8, 5))
sns.barplot(x=forest_importances, y=forest_importances.index, palette="plasma")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Gradient Boosting Feature Importance (Iris Dataset)")
plt.show()
