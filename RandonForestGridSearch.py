import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Carregar os dados do CSV
def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)
    # Remover colunas com todos os valores iguais a 0
    data = data.loc[:, (data != 0).any(axis=0)]
    X = data.drop(columns=[label_column]).values  # Recursos (features)
    y = data[label_column].values  # Rótulos (labels)
    return X, y

# Função para misturar aleatoriamente as linhas do conjunto de dados
def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Carregar e preparar os dados
csv_file = r'C:\\TCC\\Output\\ML\\TD.csv'  # Caminho para seu arquivo CSV
label_column = 'TD'  # Substitua pelo nome do campo que é para ser usado como label
X, y = load_data(csv_file, label_column)

# Misturar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

""""
Hiperparâmetros do XGBoost:
n_estimators: O número de árvores no modelo.
max_depth: A profundidade máxima de uma árvore.
min_samples_split: O número mínimo de amostras necessárias para dividir um nó.
min_samples_leaf: O número mínimo de amostras que devem estar presentes em uma folha.
"""

# Função para calcular MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para executar o treinamento e avaliação para uma combinação de hiperparâmetros
def train_and_evaluate_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mse, rmse, mae, mape

# Grid Search manual
best_mse_rf = float('inf')
best_params_rf = None
best_rmse_rf = None
best_mae_rf = None
best_mape_rf = None

for n_estimators in param_grid_rf['n_estimators']:
    for max_depth in param_grid_rf['max_depth']:
        for min_samples_split in param_grid_rf['min_samples_split']:
            for min_samples_leaf in param_grid_rf['min_samples_leaf']:
                mse_rf, rmse_rf, mae_rf, mape_rf = train_and_evaluate_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf)
                if mse_rf < best_mse_rf:
                    best_mse_rf = mse_rf
                    best_rmse_rf = rmse_rf
                    best_mae_rf = mae_rf
                    best_mape_rf = mape_rf
                    best_params_rf = (n_estimators, max_depth, min_samples_split, min_samples_leaf)

# Salvar os resultados em um arquivo CSV
results = {
    'Modelo': ['CNN'],
    'MAE': [best_mae_rf],
    'REQM': [best_rmse_rf],
    'MAPE': [best_mape_rf],
    'MSE': [best_mse_rf],
     'Best parameters': [f'n_estimators={best_params_rf[0]}, max_depth={best_params_rf[1]}, min_samples_split={best_params_rf[2]}, min_samples_leaf={best_params_rf[3]}']
}  

df_results = pd.DataFrame(results)
df_results.to_csv(r'C:\\TCC\\Output\\ML\\model_randonForest_best_params_metrics.csv', index=False)

# Imprimir os melhores hiperparâmetros e métricas
print(f'Best parameters (Random Forest): n_estimators={best_params_rf[0]}, max_depth={best_params_rf[1]}, min_samples_split={best_params_rf[2]}, min_samples_leaf={best_params_rf[3]}')
print(f'Best MSE (Random Forest): {best_mse_rf}')
print(f'Best RMSE (Random Forest): {best_rmse_rf}')
print(f'Best MAE (Random Forest): {best_mae_rf}')
print(f'Best MAPE (Random Forest): {best_mape_rf}%')
