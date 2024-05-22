import pandas as pd
import numpy as np
import warnings
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import seaborn as sb
from xgboost import XGBRegressor

# Suprimir avisos
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def get_data(train_path, test_path):
    # Obter dados de treino e teste
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def get_combined_data(train, test):
    target = train['TD']
    train.drop(['TD'], axis=1, inplace=True)
    combined = pd.concat([train, test], ignore_index=True)
    return combined, target

def get_cols_with_no_nans(df, col_type):
    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    elif col_type == 'all':
        predictors = df
    else:
        raise ValueError('Error: choose a type (num, no_num, all)')
    
    cols_with_no_nans = [col for col in predictors.columns if not df[col].isnull().any()]
    return cols_with_no_nans

def one_hot_encode(df, col_names):
    for col in col_names:
        if df[col].dtype == np.dtype('object'):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop([col], axis=1, inplace=True)
    return df

def split_combined(combined, train_len):
    train = combined.iloc[:train_len, :]
    test = combined.iloc[train_len:, :]
    return train, test

def make_submission(prediction, test_path, sub_name):
    test = pd.read_csv(test_path)
    my_submission = pd.DataFrame({'ID': test.index, 'TD': prediction})
    my_submission.to_csv(sub_name, index=False)
    print(f'A submission file has been made: {sub_name}.csv')

def remove_zero_columns(df):
    non_zero_cols = [col for col in df.columns if df[col].sum() != 0]
    return df[non_zero_cols]

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the directory.")
    latest_file = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_file)

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

# Caminhos dos arquivos
input_csv_path = r'C:\\TCC\\Output\\ML\\TD.csv'  # Substitua pelo caminho do seu arquivo CSV de entrada
train_csv_path = r'C:\\TCC\\Output\\ML\\train.csv'  # Substitua pelo caminho de saída desejado para o arquivo de treino
test_csv_path = r'C:\\TCC\\Output\\ML\\test.csv'    # Substitua pelo caminho de saída desejado para o arquivo de teste

# Chamar a função para processar o arquivo CSV com split_ratio = 0.5 para dividir igualmente
shuffle_and_split_csv(input_csv_path, train_csv_path, test_csv_path, split_ratio=0.5)

# Caminhos dos arquivos
train_data_path = r'C:\\TCC\\Output\\ML\\train.csv'
test_data_path = r'C:\\TCC\\Output\\ML\\test.csv'
checkpoint_dir = r'C:\\TCC\\Output\\ML\\Checkpoints'  # Diretório onde os checkpoints são salvos

# Load train and test data into pandas DataFrames
train_data, test_data = get_data(train_data_path, test_data_path)

# Combine train and test data to process them together
combined, target = get_combined_data(train_data, test_data)

# Remove columns with all zeros
combined = remove_zero_columns(combined)

num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print(f'Number of numerical columns with no nan values: {len(num_cols)}')
print(f'Number of non-numerical columns with no nan values: {len(cat_cols)}')

combined = combined[num_cols + cat_cols]
combined.hist(figsize=(12, 10))
# Salvar o gráfico na pasta
plt.savefig('C:\\TCC\\Output\\ML\\hist.png')

train_data_len = len(train_data)
train_data = combined[:train_data_len]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize=(15, 15))
sb.heatmap(C_mat, vmax=.8, square=True)

# Salvar o gráfico na pasta
fig.savefig('C:\\TCC\\Output\\ML\\heatmap.png')
