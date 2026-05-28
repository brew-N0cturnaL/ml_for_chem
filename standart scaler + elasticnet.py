from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

data = pd.read_csv("filtered_data2.csv")

numeric_data = data.select_dtypes(include = np.number)

x = numeric_data.drop(["Solubility", "SD", "Ocurrences", "NumSaturatedRings", "TPSA", "NumHAcceptors", "NumAliphaticRings"], axis = 1)
y = data["Solubility"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = ElasticNet(alpha = 0.001, l1_ratio= 0.5)
model.fit(x_train_scaled, y_train)

train_score = model.score(x_train_scaled, y_train)
test_score = model.score(x_test_scaled, y_test)
y_pred = model.predict(x_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

print(f"коэффициенты: {model.coef_}")
print(f"R^2 на тесте: {test_score:2f}")
