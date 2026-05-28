import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import *

data = pd.read_csv("filtered_data2.csv")

numeric_data = data.select_dtypes(include = np.number)

x = numeric_data.drop(["Solubility", "SD", "Ocurrences", "NumSaturatedRings", "TPSA", "NumHAcceptors", "NumAliphaticRings"], axis = 1)
y = data["Solubility"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

model = Ridge(alpha = 1)

grid_search = GridSearchCV( estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_search.fit(x_train_scaled, y_train)

print(f'лучшее альфа: {grid_search.best_params_["alpha"]}')

alpha = grid_search.best_params_['alpha']

best_ridge = Ridge(alpha = alpha)
best_ridge.fit(x_train_scaled, y_train)
y_pred = best_ridge.predict(x_test_scaled)
r2 = best_ridge.score(x_test_scaled, y_test)
mse = root_mean_squared_error(y_test, y_pred)

# Calculate correlation matrix
corr_matrix = numeric_data.corr()

# Extract correlation values for target variable 'A'
target_variable = 'Solubility'
target_correlations = corr_matrix[target_variable].drop(["Solubility",
                                                        "SD",
                                                        "Ocurrences",
                                                        "NumSaturatedRings",
                                                        "TPSA",
                                                        "NumHAcceptors",
                                                        "NumAliphaticRings",
                                                        "BalabanJ",
                                                        "NumRotatableBonds",
                                                        "RingCount",
                                                        "BertzCT",
                                                        "HeavyAtomCount"])

# Sort correlations
target_correlations = target_correlations.sort_values(ascending=True)

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(target_correlations.index,
               target_correlations.values, color="skyblue")

plt.title(f'Корреляция с растворимостью')
plt.xlabel('Признаки')
plt.ylabel('Коэффициент корреляции')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.6f}")

