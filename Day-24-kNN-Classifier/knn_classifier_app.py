import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset

data = pd.read_csv("data.csv")
X = data[['study_hours', 'sleep_hours']]
y = data['pass_exam']

# Step 2: Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train kNN Model

k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate Model

y_pred = model.predict(X_test_scaled)

print(f"✅ Model Trained Successfully with k={k}")
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("🧾 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title(f"Confusion Matrix (k={k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Predict New Samples

samples = [[4,7],[8,3],[5,5]]
samples_scaled = scaler.transform(samples)
predictions = model.predict(samples_scaled)

for i, (sample, pred) in enumerate(zip(samples, predictions)):
    print(f"Sample {i+1}: Study={sample[0]}, Sleep={sample[1]} → Predicted Pass={pred}")
