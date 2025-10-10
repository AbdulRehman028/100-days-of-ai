import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset

data = pd.read_csv("data.csv")

X = data[['study_hours', 'sleep_hours']]
y = data['pass_exam']

# Step 2: Split the Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Normalize Features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Create and Train Model

model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate

y_pred = model.predict(X_test_scaled)

print("✅ Model Trained Successfully!")
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Test Predictions

sample_data = scaler.transform([[4, 6], [8, 2]])
predictions = model.predict(sample_data)
print("\n🔮 Predictions:")
print("Study=4, Sleep=6 →", predictions[0])
print("Study=8, Sleep=2 →", predictions[1])