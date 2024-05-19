import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Conectar ao banco de dados SQLite
conexao = sqlite3.connect(r'C:\TCC\BD\ck\Metricas_ck.db')

# Consulta SQL para obter os dados do banco de dados
consulta_sql = "SELECT * FROM TABLE_DATASET_NORMALIZE"

# Carregando os dados em um DataFrame do Pandas
df = pd.read_sql_query(consulta_sql, conexao)
df['FILE'] = pd.to_numeric(df['FILE'], errors='coerce')

# Fechando a conexão com o banco de dados
conexao.close()

# Calculando a matriz de correlação
matriz_correlacao = df.corr()

# Escolha uma característica para visualizar a correlação com outras características
caracteristica_selecionada = 'TD_NOVA'

# Extrair correlações da característica selecionada com todas as outras características
correlacoes = matriz_correlacao[caracteristica_selecionada].drop(caracteristica_selecionada)

# Criar gráfico de bolha
plt.figure(figsize=(10, 6))
sns.scatterplot(x=correlacoes.index, y=[0]*len(correlacoes), size=correlacoes.values, sizes=(100, 500), legend=False)
plt.title(f'Correlação com a Característica: {caracteristica_selecionada}')
plt.xlabel('Características')
plt.ylabel('')  # Remover rótulo do eixo y
plt.xticks(rotation=90)  # Rotacionar rótulos do eixo x para melhor legibilidade
plt.show()