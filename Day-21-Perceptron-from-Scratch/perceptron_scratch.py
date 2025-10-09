import numpy as np
import pandas as pd

# ------------------------
# Step 1: Load Data
# ------------------------
data = pd.read_csv("data.csv")
X = data[['study_hours', 'sleep_hours']].values
y = data['pass_exam'].values

# ------------------------
# Step 2: Initialize Parameters
# ------------------------
np.random.seed(42)
weights = np.random.rand(X.shape[1])
bias = np.random.rand(1)
learning_rate = 0.01
epochs = 100

# Step 3: Define Activation (Step Function)
# ------------------------
def activation(z):
    return 1 if z >= 0 else 0

# Step 4: Training Loop
# ------------------------
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = activation(linear_output)
        error = y[i] - y_pred

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
        total_error += abs(error)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Total Error = {total_error}")

# Step 5: Final Weights
# ------------------------
print("\n✅ Training Complete!")
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")

# Step 6: Test Prediction
# ------------------------
def predict(study, sleep):
    z = np.dot(np.array([study, sleep]), weights) + bias
    return activation(z)

print("\n📊 Predictions:")
print("Study=4, Sleep=6 →", predict(4, 6))
print("Study=8, Sleep=2 →", predict(8, 2))
