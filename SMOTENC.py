import pandas as pd
from imblearn.over_sampling import ADASYN
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Função para carregar os dados do CSV
def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)
    X = data.drop(columns=[label_column])
    y = data[label_column]
    return X, y

# Função para aplicar ADASYN e salvar os dados balanceados em um novo CSV
def apply_adasyn(csv_file, label_column, output_file):
    # Carregar os dados
    X, y = load_data(csv_file, label_column)
    
    # Aplicar ADASYN
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X, y)
    
    # Verificar a distribuição das classes após o balanceamento
    print(f'Distribuição das classes antes do ADASYN: {Counter(y)}')
    print(f'Distribuição das classes após o ADASYN: {Counter(y_res)}')
    
    # Criar um DataFrame com os dados balanceados
    balanced_data = pd.DataFrame(X_res, columns=X.columns)
    balanced_data[label_column] = y_res
    
    # Salvar o DataFrame balanceado em um novo arquivo CSV
    balanced_data.to_csv(output_file, index=False)
    print(f'Dados balanceados salvos em: {output_file}')

def apply_random_oversampler_and_adasyn(csv_file, label_column, output_file):
    # Carregar os dados
    X, y = load_data(csv_file, label_column)
    
    # Aplicar RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    
    # Aplicar ADASYN nos dados resultantes
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_res, y_res)
    
    # Verificar a distribuição das classes após o balanceamento
    print(f'Distribuição das classes antes do balanceamento: {Counter(y)}')
    print(f'Distribuição das classes após o balanceamento: {Counter(y_res)}')
    
    # Criar um DataFrame com os dados balanceados
    balanced_data = pd.DataFrame(X_res, columns=X.columns)
    balanced_data[label_column] = y_res
    
    # Salvar o DataFrame balanceado em um novo arquivo CSV
    balanced_data.to_csv(output_file, index=False)
    print(f'Dados balanceados salvos em: {output_file}')    

# Parâmetros
csv_file = r'C:\\TCC\\Output\\ML\\TD.csv'  # Caminho para o arquivo CSV de entrada
label_column = 'TD'  # Nome da coluna de rótulo (label)
output_file = r'C:\\TCC\\Output\\ML\\TD_BALANCED.csv'  # Caminho para o arquivo CSV de saída

# Executar o balanceamento com ADASYN
#apply_adasyn(csv_file, label_column, output_file)

# Executar o balanceamento com RandomOverSampler e ADASYN
apply_random_oversampler_and_adasyn(csv_file, label_column, output_file)
