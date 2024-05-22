import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
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

param_grid_xgb2 = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.01],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'min_child_weight': [1]
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

""""
Hiperparâmetros do XGBoost:
n_estimators: O número de árvores no modelo.
max_depth: A profundidade máxima de uma árvore.
learning_rate: A taxa de aprendizado (ou eta), que diminui o impacto de cada árvore.
subsample: A fração das amostras usadas para treinar cada árvore.
colsample_bytree: A fração das features a serem usadas para cada árvore.
min_child_weight: O peso mínimo de uma folha.
"""

# Função para calcular MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para executar o treinamento e avaliação para uma combinação de hiperparâmetros
def train_and_evaluate_xgb(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight):
    xgb_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                             random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mse, rmse, mae, mape

# Grid Search manual
best_mse_xgb = float('inf')
best_params_xgb = None
best_rmse_xgb = None
best_mae_xgb = None
best_mape_xgb = None

for n_estimators in param_grid_xgb['n_estimators']:
    for max_depth in param_grid_xgb['max_depth']:
        for learning_rate in param_grid_xgb['learning_rate']:
            for subsample in param_grid_xgb['subsample']:
                for colsample_bytree in param_grid_xgb['colsample_bytree']:
                    for min_child_weight in param_grid_xgb['min_child_weight']:
                        mse_xgb, rmse_xgb, mae_xgb, mape_xgb = train_and_evaluate_xgb(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight)
                        if mse_xgb < best_mse_xgb:
                            best_mse_xgb = mse_xgb
                            best_rmse_xgb = rmse_xgb
                            best_mae_xgb = mae_xgb
                            best_mape_xgb = mape_xgb
                            best_params_xgb = (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight)

# Salvar os resultados em um arquivo CSV
results = {
    'Modelo': ['GXBoost'],
    'MAE': [best_mae_xgb],
    'REQM': [best_rmse_xgb],
    'MAPE': [best_mape_xgb],
    'MSE': [best_mse_xgb],
     'Best parameters': [f'n_estimators={best_params_xgb[0]}, max_depth={best_params_xgb[1]}, learning_rate={best_params_xgb[2]}, subsample={best_params_xgb[3]}, colsample_bytree={best_params_xgb[4]}, min_child_weight={best_params_xgb[5]}']
}  

df_results = pd.DataFrame(results)
df_results.to_csv(r'C:\\TCC\\Output\\ML\\model_xboost_best_params_metrics.csv', index=False)


# Imprimir os melhores hiperparâmetros e métricas
print(f'Best parameters (XGBoost): n_estimators={best_params_xgb[0]}, max_depth={best_params_xgb[1]}, learning_rate={best_params_xgb[2]}, subsample={best_params_xgb[3]}, colsample_bytree={best_params_xgb[4]}, min_child_weight={best_params_xgb[5]}')
print(f'Best MSE (XGBoost): {best_mse_xgb}')
print(f'Best RMSE (XGBoost): {best_rmse_xgb}')
print(f'Best MAE (XGBoost): {best_mae_xgb}')
print(f'Best MAPE (XGBoost): {best_mape_xgb}%')
