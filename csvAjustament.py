import pandas as pd
import numpy as np

def shuffle_and_split_csv(input_csv_path, train_csv_path, test_csv_path, split_ratio=0.5, random_state=42):
    # Carregar os dados do CSV
    df = pd.read_csv(input_csv_path)
    
    # Fazer shuffle no dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Determinar o ponto de corte para dividir em treino e teste
    train_size = int(split_ratio * len(df_shuffled))
    
    # Dividir o dataframe
    train_df = df_shuffled.iloc[:train_size]
    test_df = df_shuffled.iloc[train_size:]
    
    # Salvar os dataframes em arquivos CSV separados, mantendo os cabeçalhos
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f'Arquivo de treino salvo como {train_csv_path}')
    print(f'Arquivo de teste salvo como {test_csv_path}')

# Caminhos dos arquivos
input_csv_path = r'C:\\TCC\\Output\\ML\\TD.csv'  # Substitua pelo caminho do seu arquivo CSV de entrada
train_csv_path = r'C:\\TCC\\Output\\ML\\train.csv'  # Substitua pelo caminho de saída desejado para o arquivo de treino
test_csv_path = r'C:\\TCC\\Output\\ML\\test.csv'    # Substitua pelo caminho de saída desejado para o arquivo de teste

# Chamar a função para processar o arquivo CSV com split_ratio = 0.5 para dividir igualmente
shuffle_and_split_csv(input_csv_path, train_csv_path, test_csv_path, split_ratio=0.5)
