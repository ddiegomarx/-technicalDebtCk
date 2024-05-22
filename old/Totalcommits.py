
import os
import shutil
import subprocess
import sqlite3
import csv
import pandas as pd
import git
import threading

def is_xml_file(filename):
    return filename.lower().endswith('.xml')

def PercorreCommit(rep, caminho):
    # Diretório local onde o repositório será clonado
    destination_path  = caminho

    # Diretório resultado ck
    destinationck_path = destination_path + "_ck\\"

    # URL do repositório que você deseja clonar
    repo_path  = rep
    # LInha cmd Execução CK
    comando = "java -jar ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar " + destination_path + " true 0 True "

    # Clone o repositório
    git.Repo.clone_from(repo_path, destination_path)

    # Acessar o repositório local
    repo = git.Repo(destination_path)

    # Criar uma pasta temporária para aplicar as alterações
    temp_folder = destination_path
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    total_commits = len(list(repo.iter_commits()))
    print("Total de commit do " + rows["projectID"] +  str(total_commits))

    repo.close()

# Conectar ao banco de dados ou criar um novo arquivo de banco de dados
conn = sqlite3.connect("C:\Denis\TCC\TechnicalDebtDataset_20200606.db")

# Criar um cursor
cursor = conn.cursor()

# Executar um comando SQL para consultar os dados na tabela
select_query = "SELECT PROJECTS.gitLink, PROJECTS.projectID FROM PROJECTS ORDER BY 2 DESC "
cursor.execute(select_query)
rows = cursor.fetchall()
data = pd.read_sql(select_query, conn)

for index, rows in data.iterrows():
    repo_url = rows["gitLink"]
    local_dir = "C:\\Denis\\Projetos\\" + rows["projectID"]
    PercorreCommit(repo_url, local_dir)
   