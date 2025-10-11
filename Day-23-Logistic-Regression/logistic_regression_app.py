import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("data.csv")

X = data[['study_hours', 'sleep_hours']]
y = data['pass_exam']

# -----------------------------
# Step 2: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("✅ Model Trained Successfully!")
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Step 6: Predict New Samples
# -----------------------------
sample_data = scaler.transform([[4, 7], [8, 3]])
predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

print("\n🔮 Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i+1}: Predicted={pred}, Probability={prob}")
