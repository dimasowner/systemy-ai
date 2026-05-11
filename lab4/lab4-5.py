import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X, y)
y_pred_simple = lin_reg_simple.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"Первинна ознака X[0]: {X[0]}")
print(f"Перетворена ознака X_poly[0]: {X_poly[0]}")

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

print(f"\nВільний член (intercept): {lin_reg_poly.intercept_}")
print(f"Коефіцієнти моделі (coef): {lin_reg_poly.coef_}")

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg_poly.predict(X_new_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вхідні дані (крапки)')
plt.plot(X_new, y_new, color='red', linewidth=3, label='Поліноміальна регресія (deg=2)')
plt.plot(X, y_pred_simple, color='green', linestyle='--', label='Лінійна регресія')

plt.title('Завдання 2.5: Поліноміальна регресія (Варіант 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

y_poly_pred = lin_reg_poly.predict(X_poly)
print(f"\nМетрики якості поліноміальної моделі:")
print(f"MSE: {mean_squared_error(y, y_poly_pred):.4f}")
print(f"R2 Score: {r2_score(y, y_poly_pred):.4f}")