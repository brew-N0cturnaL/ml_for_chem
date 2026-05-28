from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from matplotlib import *
import matplotlib.pyplot as plt

data = pd.read_csv("filtered_data2.csv")

numeric_data = data.select_dtypes(include = np.number)


x = numeric_data.drop(["Solubility", "SD", "Ocurrences"], axis = 1)
x.to_excel("numeric_data.xlsx", index = False)
y = data["Solubility"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

train_score = model.score(x_train_scaled, y_train)
test_score = model.score(x_test_scaled, y_test)
y_pred = model.predict(x_test_scaled)
mse = root_mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

print(f"R^2 на тесте: {test_score:2f}")
print(f"R^2 на обучении: {train_score:2f}")
print(f"Веса: {model.coef_}")

plt.figure(figsize = (8,5))
plt.grid()
plt.scatter(y_test, y_pred, alpha = 0.5, color='blue', label='Данные')
plt.plot((-12, 5),(-12, 5),  color='red', label='y = x')
plt.title(f'Реальная vs предсказанная растворимость\nТестовый $R^2$ = {test_score:2f}', fontsize=14)
plt.xlabel('Реальные значения', fontsize=12)
plt.ylabel('Пресдказанные значения', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()