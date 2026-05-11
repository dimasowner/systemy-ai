import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Зчитування даних з існуючого файлу
# Переконайся, що файл data_singlevar_regr_2.txt лежить у тій же папці, що і цей скрипт
data = np.genfromtxt('data_regr_2.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# 2. Навчання моделі
model = LinearRegression()
model.fit(X, y)

# 3. Прогнозування
y_pred = model.predict(X)

# 4. Вивід результатів у консоль
print(f"Коефіцієнт (Slope): {model.coef_[0]:.4f}")
print(f"Вільний член (Intercept): {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"R2 Score: {r2_score(y, y_pred):.4f}")

# 5. Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вхідні дані')
plt.plot(X, y_pred, color='red', linewidth=2, label='Регресійна лінія')
plt.title('Завдання 2.2: Регресія однієї змінної')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()