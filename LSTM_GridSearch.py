import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Função para construir o modelo LSTM
def build_lstm_model(units=50, activation='relu', optimizer='adam', input_shape=(10, 1)):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation, input_shape=input_shape))
    model.add(Dense(1))  # Para regressão
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model

# Função para calcular o RMSE
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Função para calcular o MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Carregar e preparar os dados
csv_file = r'C:\\TCC\\Output\\ML\\TD.csv'  # Caminho para seu arquivo CSV
label_column = 'TD'  # Substitua pelo nome do campo que é para ser usado como label
X, y = load_data(csv_file, label_column)

# Misturar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

# Redimensionar os dados para a LSTM (adicionar uma dimensão para passos temporais)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir hiperparâmetros do modelo
input_shape = (X_train.shape[1], 1)

# Definir o grid de parâmetros manualmente
param_grid2 = {
    'units': [32],
    'activation': ['sigmoid'],
    'optimizer': ['SGD'],
    'epochs': [8],
    'batch_size': [8]
}

param_grid = {
    'units': [32, 64, 128, 256, 512],
    'activation': ['relu', 'sigmoid'],
    'optimizer': ['adam', 'RMSprop', 'SGD'],
    'epochs': [10, 20, 30],
    'batch_size': [8, 16, 32]
}


# Função para executar o treinamento e avaliação para uma combinação de hiperparâmetros
def train_and_evaluate(units, activation, optimizer, epochs, batch_size):
    model = build_lstm_model(units=units, activation=activation, optimizer=optimizer, input_shape=input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test)
    
    # Verificar se y_pred contém valores não finitos
    if not np.all(np.isfinite(y_pred)):
        return float('inf')
    
    error = mean_squared_error(y_test, y_pred)
    return error

# Grid Search manual
best_score = float('inf')
best_params = None

for units in param_grid['units']:
    for activation in param_grid['activation']:
        for optimizer in param_grid['optimizer']:
            for epochs in param_grid['epochs']:
                for batch_size in param_grid['batch_size']:
                    score = train_and_evaluate(units, activation, optimizer, epochs, batch_size)
                    print(f'Tested params - units: {units}, activation: {activation}, optimizer: {optimizer}, epochs: {epochs}, batch_size: {batch_size}, MSE: {score}')
                    if score < best_score:
                        best_score = score
                        best_params = (units, activation, optimizer, epochs, batch_size)

# Imprimir os melhores hiperparâmetros
print(f'Best parameters: units={best_params[0]}, activation={best_params[1]}, optimizer={best_params[2]}, epochs={best_params[3]}, batch_size={best_params[4]}')
print(f'Best MSE: {best_score}')

# Avaliar o modelo final com os melhores parâmetros
best_model = build_lstm_model(units=best_params[0], activation=best_params[1], optimizer=best_params[2], input_shape=input_shape)
best_model.fit(X_train, y_train, epochs=best_params[3], batch_size=best_params[4], verbose=0)
y_pred = best_model.predict(X_test)

# Calcular e imprimir as métricas RMSE, MAE e MAPE

rmse_value = rmse(y_test, y_pred)
mae_value = mean_absolute_error(y_test, y_pred)
mape_value = mape(y_test, y_pred)

# Salvar os resultados em um arquivo CSV
results = {
    'Modelo': ['LSTM'],
    'MAE': [mae_value],
    'REQM': [rmse_value],
    'MAPE': [mape_value],
    'MSE': [best_score],
    'Best parameters': [f'units={best_params[0]}, activation={best_params[1]}, optimizer={best_params[2]}, epochs={best_params[3]}, batch_size={best_params[4]}']
}  

df_results = pd.DataFrame(results)
df_results.to_csv(r'C:\\TCC\\Output\\ML\\model_LSTM_best_params_metrics.csv', index=False)

print(f'LSTM validation RMSE: {rmse_value}')
print(f'LSTM validation MAE: {mae_value}')
print(f'LSTM validation MAPE: {mape_value}')
