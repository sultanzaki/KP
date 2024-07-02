import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import pickle

data = pd.read_csv('dataset.csv')
df = data.drop(['Provinsi', 'Tahun'], axis=1)

X = df.drop(columns='Produksi')
y = df[['Produksi']]

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

gb = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train.values.ravel())
best_gb_model = grid_search.best_estimator_

voting_model = VotingRegressor(estimators=[
    ('lr', LinearRegression()),
    ('ridge', best_model),
    ('gb', best_gb_model)
])
voting_model.fit(x_train, y_train.values.ravel())

y_pred_ensemble = voting_model.predict(x_test)
print(f"Model Terbaik R^2: {r2_score(y_test, y_pred_ensemble)}")

def predict_production(input_data, poly, scaler, model):
    input_df = pd.DataFrame([input_data])
    input_poly = poly.transform(input_df)
    input_scaled = scaler.transform(input_poly)
    prediction = model.predict(input_scaled)
    return prediction[0]

input_data = {
    'Luas Panen': 330000.00,
    'Curah hujan': 1600.00,
    'Kelembapan': 82.50,
    'Suhu rata-rata': 26.50
}

prediction = predict_production(input_data, poly, sc, voting_model)
print(f"Prediksi Produksi: {prediction}")