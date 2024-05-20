import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar os dados do CSV
def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)
    X = data.drop(columns=[label_column]).values  # Recursos (features)
    y = data[label_column].values  # Rótulos (labels)
    # Ajustar rótulos menores que 0 para 0
    y = np.where(y < 0, 0, y)
    return X, y

# Função para misturar aleatoriamente as linhas do conjunto de dados
def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Construir modelo CNN para dados tabulares
def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Carregar e preparar os dados
csv_file = r'C:\\TCC\\Output\\ML\\TD_BALANCED.csv'
label_column = 'TD'  # Substitua 'your_label_column' pelo nome do campo que é para ser usado como label
X, y = load_data(csv_file, label_column)

# Misturar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar os rótulos para começar de 0
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)
# Normalizar os dados (opcional, mas recomendado para redes neurais)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Redimensionar os dados para a CNN (adicionar uma dimensão para canais)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Definir hiperparâmetros do modelo
input_shape = (X_train.shape[1], 1)
num_classes = len(np.unique(y_train))

# Construir e compilar o modelo CNN
model = build_cnn_model(input_shape, num_classes)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
