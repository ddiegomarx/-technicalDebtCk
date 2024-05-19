import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Conectar ao banco de dados ou criar um novo arquivo de banco de dados
conn = sqlite3.connect("C:\DB\TechnicalDebtDataset_20200606.db")

# Criar um cursor
cursor = conn.cursor()

# Executar um comando SQL para consultar os dados na tabela
select_query = "SELECT TD_1000.* FROM TD_1000"
cursor.execute(select_query)
rows = cursor.fetchall()
data = pd.read_sql(select_query, conn)

# Exibir os resultados da consulta
# for row in rows:
#  print(row)

# Dividir os dados em recursos (X) e rótulos (y)
y = data['tipo']
X = data.drop(columns=['tipo'])  # Certifique-se de ajustar as colunas conforme necessário
#X = data_encoded.drop(columns=['status'])

X_encoded = pd.get_dummies(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Criar o modelo de árvore de decisão
#model = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=10000, random_state=42)

# Treinar o modelo com os dados de treinamento
#model.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Fazer previsões com o conjunto de teste
#predictions = model.predict(X_test)

# Avaliar a precisão do modelo
#accuracy = accuracy_score(y_test, predictions)
#print(f'Accuracy: {accuracy:.2f}')
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)