import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras_tuner as kt

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

# Função para construir o modelo LSTM com hiperparâmetros ajustáveis
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   activation='sigmoid', input_shape=input_shape))
    model.add(Dense(1))  # Para regressão
    model.compile(loss='mean_squared_error', 
                  optimizer=hp.Choice('optimizer', values=['adam', 'RMSprop', 'SGD']),
                  metrics=['mae'])
    return model

# Função para calcular o RMSE
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Função para calcular o MAPE
def mape(y_true, y_pred, eps=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


# Carregar e preparar os dados
csv_file = r'C:\\TCC\\Output\\ML\\TD.csv'  # Caminho para seu arquivo CSV
label_column = 'TD'  # Substitua pelo nome do campo que é para ser usado como label
X, y = load_data(csv_file, label_column)

# Misturar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Redimensionar os dados para a LSTM (adicionar uma dimensão para passos temporais)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir hiperparâmetros do modelo
input_shape = (X_train.shape[1], 1)

# Usar Keras Tuner para encontrar os melhores hiperparâmetros
tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_mae',
    max_trials=10,  # Ajuste conforme necessário
    executions_per_trial=3,
    directory='my_dir',
    project_name='lstm_tuning'
)

# Realizar a busca de hiperparâmetros
tuner.search(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the LSTM layer is {best_hps.get('units')} 
and the best optimizer is {best_hps.get('optimizer')}.
""")

# Construir o modelo com os melhores hiperparâmetros
model = tuner.hypermodel.build(best_hps)

# Treinar o modelo final
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, mae = model.evaluate(X_test, y_test)

# Prever os valores de y
y_pred = model.predict(X_test)

# Calcular e imprimir o RMSE e o MAPE
print(f'LSTM validation MAE: {mae}')
print(f'LSTM validation RMSE: {rmse(y_test, y_pred)}')
print(f'LSTM validation MAPE: {mape(y_test, y_pred)}')
