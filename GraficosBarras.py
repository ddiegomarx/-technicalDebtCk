import pandas as pd
import matplotlib.pyplot as plt

def gerar_grafico_barras(df, metrica):
    # Extrair os modelos e os valores da métrica
    modelos = df['Modelo']
    valores = df[metrica]
    
    # Criar o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(modelos, valores)
    
    # Adicionar título e rótulos dos eixos
    plt.title(f'Gráfico de Barras - {metrica}')
    plt.xlabel('Modelos')
    plt.ylabel(metrica)
    
    # Exibir o gráfico
    plt.xticks(rotation=45)
    plt.show()

# Carregar o arquivo CSV
csv_file = r'C:\\TCC\\Output\\ML\\METRICAS.csv' 
df = pd.read_csv(csv_file)

# Gerar gráficos de barras para cada métrica
metricas = ['MAE', 'REQM', 'MAPE', 'MSE']
for metrica in metricas:
    gerar_grafico_barras(df, metrica)

