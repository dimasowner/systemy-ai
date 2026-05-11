import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Параметри моделі:")
print(f"Коефіцієнти регресії (regr.coef_):\n{regr.coef_}")
print(f"Вільний член (regr.intercept_): {regr.intercept_:.2f}")

print("\nПоказники якості:")
print(f"R2 score: {r2_score(ytest, ypred):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(ytest, ypred):.2f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(ytest, ypred):.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), label='Передбачення моделі')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальна лінія')
ax.set_xlabel('Виміряно (Actual Values)')
ax.set_ylabel('Передбачено (Predicted Values)')
ax.set_title('Завдання 2.4: Порівняння виміряних та передбачених значень')
ax.legend()
plt.grid(True)
plt.show()