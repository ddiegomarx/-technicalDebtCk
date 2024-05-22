import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
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

# Redimensionar os dados para o formato esperado pela CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Definir o grid de parâmetros manualmente
param_grid_cnn2 = {
    'filters': [32],
    'kernel_size': [3],
    'pool_size': [2],
    'units': [50],
    'dropout_rate': [0.2],
    'batch_size': [32],
    'epochs': [10],
    'learning_rate': [0.01]
}

param_grid_cnn = {
    'filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'pool_size': [2, 3],
    'units': [50, 100, 150],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [32, 64, 128],
    'epochs': [10, 50, 100],
    'learning_rate': [0.001, 0.01, 0.1]
}

"""
Explicação dos Hiperparâmetros:
ilters: O número de filtros nas camadas convolucionais.
kernel_size: O tamanho do kernel (filtro) nas camadas convolucionais.
pool_size: O tamanho do pool nas camadas de pooling.
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

# Função para criar e treinar o modelo CNN
def train_and_evaluate_cnn(filters, kernel_size, pool_size, units, dropout_rate, batch_size, epochs, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
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
best_mse_cnn = float('inf')
best_params_cnn = None
best_rmse_cnn = None
best_mae_cnn = None
best_mape_cnn = None

for filters in param_grid_cnn['filters']:
    for kernel_size in param_grid_cnn['kernel_size']:
        for pool_size in param_grid_cnn['pool_size']:
            for units in param_grid_cnn['units']:
                for dropout_rate in param_grid_cnn['dropout_rate']:
                    for batch_size in param_grid_cnn['batch_size']:
                        for epochs in param_grid_cnn['epochs']:
                            for learning_rate in param_grid_cnn['learning_rate']:
                                mse_cnn, rmse_cnn, mae_cnn, mape_cnn = train_and_evaluate_cnn(filters, kernel_size, pool_size, units, dropout_rate, batch_size, epochs, learning_rate)
                                if mse_cnn < best_mse_cnn:
                                    best_mse_cnn = mse_cnn
                                    best_rmse_cnn = rmse_cnn
                                    best_mae_cnn = mae_cnn
                                    best_mape_cnn = mape_cnn
                                    best_params_cnn = (filters, kernel_size, pool_size, units, dropout_rate, batch_size, epochs, learning_rate)

results = {
        'Modelo': ['CNN'],
        'MAE': [best_mae_cnn],
        'REQM': [best_rmse_cnn],
        'MAPE': [best_mape_cnn],
        'MSE': [best_mse_cnn],
        'Best parameters': [f'filters={best_params_cnn[0]}, kernel_size={best_params_cnn[1]}, pool_size={best_params_cnn[2]}, units={best_params_cnn[3]}, dropout_rate={best_params_cnn[4]}, batch_size={best_params_cnn[5]}, epochs={best_params_cnn[6]}, learning_rate={best_params_cnn[7]}']
    }

df_results = pd.DataFrame(results)
df_results.to_csv(r'C:\\TCC\\Output\\ML\\model_cnn_best_params_metrics.csv', index=False)

# Imprimir os melhores hiperparâmetros e métricas
print(f'Best parameters (CNN): filters={best_params_cnn[0]}, kernel_size={best_params_cnn[1]}, pool_size={best_params_cnn[2]}, units={best_params_cnn[3]}, dropout_rate={best_params_cnn[4]}, batch_size={best_params_cnn[5]}, epochs={best_params_cnn[6]}, learning_rate={best_params_cnn[7]}')
print(f'Best MSE (CNN): {best_mse_cnn}')
print(f'Best RMSE (CNN): {best_rmse_cnn}')
print(f'Best MAE (CNN): {best_mae_cnn}')
print(f'Best MAPE (CNN): {best_mape_cnn}%')
