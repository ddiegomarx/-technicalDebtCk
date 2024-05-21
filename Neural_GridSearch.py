import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
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

# Definir o grid de parâmetros manualmente
param_grid_nn = {
    'units': [50, 100, 150],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [32, 64, 128],
    'epochs': [10, 50, 100],
    'learning_rate': [0.001, 0.01, 0.1]
}

""""
Explicação dos Hiperparâmetros:
units: O número de unidades (neurônios) nas camadas densas.
dropout_rate: A fração das unidades a serem descartadas durante o treinamento.
batch_size: O número de amostras por gradiente de atualização.
epochs: O número de épocas para treinar o modelo.
learning_rate: A taxa de aprendizado para o otimizador.
"""

# Função para calcular MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para criar e treinar o modelo
def train_and_evaluate_nn(units, dropout_rate, batch_size, epochs, learning_rate):
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return mse, rmse, mae, mape

# Grid Search manual
best_mse_nn = float('inf')
best_params_nn = None
best_rmse_nn = None
best_mae_nn = None
best_mape_nn = None

for units in param_grid_nn['units']:
    for dropout_rate in param_grid_nn['dropout_rate']:
        for batch_size in param_grid_nn['batch_size']:
            for epochs in param_grid_nn['epochs']:
                for learning_rate in param_grid_nn['learning_rate']:
                    mse_nn, rmse_nn, mae_nn, mape_nn = train_and_evaluate_nn(units, dropout_rate, batch_size, epochs, learning_rate)
                    if mse_nn < best_mse_nn:
                        best_mse_nn = mse_nn
                        best_rmse_nn = rmse_nn
                        best_mae_nn = mae_nn
                        best_mape_nn = mape_nn
                        best_params_nn = (units, dropout_rate, batch_size, epochs, learning_rate)

# Imprimir os melhores hiperparâmetros e métricas
print(f'Best parameters (Neural Network): units={best_params_nn[0]}, dropout_rate={best_params_nn[1]}, batch_size={best_params_nn[2]}, epochs={best_params_nn[3]}, learning_rate={best_params_nn[4]}')
print(f'Best MSE (Neural Network): {best_mse_nn}')
print(f'Best RMSE (Neural Network): {best_rmse_nn}')
print(f'Best MAE (Neural Network): {best_mae_nn}')
print(f'Best MAPE (Neural Network): {best_mape_nn}%')
