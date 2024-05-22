
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

    contador = 0
    # Percorrer todos os commits no repositório
    for commit in repo.iter_commits():
        commit_hash = commit.hexsha

        # Verificar se o commit afeta a pasta de destino
        affected_files = [change.a_path for change in commit.diff().iter_change_type('M')]
        for file in affected_files:
            # Verificar se o arquivo afeta a pasta de destino
            if file.startswith('src/') and not(is_xml_file(file)):
                # Obter o conteúdo do arquivo no commit
                blob = commit.tree[file]
                file_content = blob.data_stream.read()

                # Determinar o caminho relativo do arquivo
                relative_file_path = file.replace(destination_path, '')

                # Criar o caminho completo para o arquivo na pasta temporária
                temp_file_path = os.path.join(temp_folder, relative_file_path)

                # Certificar-se de que a pasta temporária exista
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Escrever o conteúdo do arquivo na pasta temporária
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_content)
            
        #Cria pasta resultado ck
        if not os.path.exists(destinationck_path):
            os.mkdir(destinationck_path) 
    
        contador += 1                  
        print(contador)
    # Fechar o repositório após a conclusão 
    print(rows["projectID"])
    print("Total de commits iterados:", contador)
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
   