import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 - X**2 + 2 * X + np.random.randn(100, 1) * 3  # cubic relationship + noise

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print("📊 Linear Regression R²:", r2_score(y, y_pred_linear))
print("📊 Polynomial Regression R²:", r2_score(y, y_pred_poly))

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data (noisy cubic)")
plt.plot(X, y_pred_linear, color="red", label="Linear Regression")
plt.plot(X, y_pred_poly, color="green", linewidth=2, label="Polynomial Regression (deg=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression vs Linear Regression")
plt.legend()
plt.show()