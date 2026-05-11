import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = np.genfromtxt('data_singlevar_regr.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]             

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(f"Коефіцієнт (нахил): {model.coef_[0]:.4f}")
print(f"Вільний член (intercept): {model.intercept_:.4f}")
print(f"Середньоквадратична помилка (MSE): {mean_squared_error(y, y_pred):.4f}")
print(f"Середня абсолютна помилка (MAE): {mean_absolute_error(y, y_pred):.4f}")
print(f"Коефіцієнт детермінації (R2): {r2_score(y, y_pred):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вхідні дані')
plt.plot(X, y_pred, color='red', linewidth=2, label='Регресійна лінія')
plt.title('Модель регресії однієї змінної')
plt.xlabel('Вхідна змінна (X)')
plt.ylabel('Цільова змінна (y)')
plt.legend()
plt.grid(True)
plt.show()