import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def calcular_estatisticas(coluna):
    media = coluna.mean()
    desvio_padrao = coluna.std()
    minimo = coluna.min()
    quartil_inferior = coluna.quantile(0.25)
    mediana = coluna.median()
    quartil_superior = coluna.quantile(0.75)
    maximo = coluna.max()
    distorcao = skew(coluna)
    curtose = kurtosis(coluna)
    
    return pd.Series({
        'Media': media,
        'Desvio padrão': desvio_padrao,
        'Mínimo': minimo,
        'Quartil inferior': quartil_inferior,
        'Mediana': mediana,
        'Quartil superior': quartil_superior,
        'Máximo': maximo,
        'Skewness': distorcao,
        'Kurtosis': curtose
    })

def gerar_estatisticas_csv(nome_arquivo_entrada, nome_arquivo_saida):
    # Carregar o arquivo CSV
    df = pd.read_csv(nome_arquivo_entrada)
    
    # Calcular estatísticas para cada coluna
    estatisticas = df.apply(calcular_estatisticas)
    
    # Transpor o DataFrame para que as colunas se tornem linhas
    estatisticas_transpostas = estatisticas.T
    
    # Arredondar os valores para no máximo 2 casas decimais
    estatisticas_transpostas = estatisticas_transpostas.round(2)
    
    # Salvar as estatísticas em um arquivo CSV
    estatisticas_transpostas.to_csv(nome_arquivo_saida)

# Exemplo de uso:
arquivo_entrada = r'C:\\TCC\\Output\\ML\\TD.csv'
arquivo_saida = r'C:\\TCC\\Output\\ML\\estatisticas.csv'
gerar_estatisticas_csv(arquivo_entrada, arquivo_saida)
