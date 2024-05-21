import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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

# Construir modelo LSTM para dados tabulares
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Para regressão
    return model

# Função para calcular o RMSE
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Função para calcular o MAPE
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Carregar e preparar os dados
csv_file = r'C:\\TCC\\Output\\ML\\TD.csv'  # Caminho para seu arquivo CSV
label_column = 'TD'  # Substitua pelo nome do campo que é para ser usado como label
X, y = load_data(csv_file, label_column)

# Misturar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Redimensionar os dados para a LSTM (adicionar uma dimensão para passos temporais)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir hiperparâmetros do modelo
input_shape = (X_train.shape[1], 1)

# Construir e compilar o modelo LSTM
model = build_lstm_model(input_shape)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Treinar o modelo
#model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test),  validation_split=0.50)
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, mae = model.evaluate(X_test, y_test)

# Prever os valores de y
y_pred = model.predict(X_test)

# Calcular e imprimir o RMSE e o MAPE
print(f'LSTM validation MAE: {mae}')
print(f'LSTM validation RMSE: {rmse(y_test, y_pred)}')
print(f'LSTM validation MAPE: {mape(y_test, y_pred)}')
