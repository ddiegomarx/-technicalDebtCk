import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Conectando ao banco de dados SQLite
conexao = sqlite3.connect('C:\TCC\BD\ck\Metricas_ck.db')

# Consulta SQL para obter os dados do banco de dados
consulta_sql = "SELECT * FROM TABLE_DATASET_NORMALIZE"

# Carregando os dados em um DataFrame do Pandas
df = pd.read_sql_query(consulta_sql, conexao)
df['FILE'] = pd.to_numeric(df['FILE'], errors='coerce')


# Fechando a conexão com o banco de dados
conexao.close()

# Calculando a matriz de correlação
matriz_correlacao = df.corr()

# Criando o gráfico de correlação com Seaborn
plt.figure(figsize=(100, 60))
sns.heatmap(matriz_correlacao, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title('Gráfico de Correlação')
plt.show()