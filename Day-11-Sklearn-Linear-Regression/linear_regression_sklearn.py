import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    data = pd.read_csv("sample_data.csv")
    X = data[["Feature"]].values
    y = data["Target"].values
    print("📂 Loaded dataset from sample_data.csv")
except FileNotFoundError:
    print("⚠️ sample_data.csv not found, generating synthetic dataset...")
    # Generate synthetic linear data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Coefficient (slope): {model.coef_[0]:.2f}")
print(f"✅ Model Intercept: {model.intercept_:.2f}")
print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📊 R² Score: {r2:.2f}")

plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression with Scikit-Learn")
plt.legend()
plt.show()