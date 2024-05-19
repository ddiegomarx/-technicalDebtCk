import sqlite3
import git
import os
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
select_query = "SELECT PROJECTS.gitLink, PROJECTS.projectID FROM PROJECTS "
cursor.execute(select_query)
rows = cursor.fetchall()
data = pd.read_sql(select_query, conn)

for index, rows in data.iterrows():
    repo_url = rows["gitLink"]
    local_dir = "C:\Projetos\ckAnalizer\\" + rows["projectID"]
    arquivos  = os.listdir(local_dir)
    if (not (arquivos)) or (not(os.path.exists(local_dir))):
        git.Repo.clone_from(repo_url, local_dir)
        print("Repositório clonado com sucesso em:", local_dir)