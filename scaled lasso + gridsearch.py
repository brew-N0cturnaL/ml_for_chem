from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv("filtered_data2.csv")

numeric_data = data.select_dtypes(include = np.number)

x = numeric_data.drop(["Solubility", "SD", "Ocurrences", "NumSaturatedRings", "TPSA", "NumHAcceptors", "NumAliphaticRings"], axis = 1)
y = data["Solubility"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lasso = Lasso(max_iter=10000)

param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 10, 100] }

grid_search = GridSearchCV(estimator = lasso, param_grid = param_grid, cv=5, n_jobs = -1)
grid_search.fit(x_train_scaled, y_train)

print(f"лучшее alpha: {grid_search.best_params_['alpha']}")

best_lasso = grid_search.best_estimator_
best_lasso.fit(x_train_scaled, y_train)
y_pred = best_lasso.predict(x_test_scaled)
r2 = best_lasso.score(x_test_scaled, y_test)
mse = root_mean_squared_error(y_test, y_pred)
print(f"R^2: {r2:.6f}")
print(f"MSE на тесте: {mse:.4f}")