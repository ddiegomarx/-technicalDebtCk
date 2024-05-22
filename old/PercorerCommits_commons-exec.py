import git
import os
import shutil
import subprocess
import sqlite3
import csv
import pandas as pd
import threading


def is_xml_file(file_path):
    # Obtém a extensão do arquivo
    file_extension = os.path.splitext(file_path)[1]

    # Compara a extensão com ".xml" (ignorando letras maiúsculas/minúsculas)
    return file_extension.lower() == ".xml"

def InsertBD(hashCode, caminho):
    # Nome do arquivo CSV
    arquivo_csv = caminho + "\\class.csv"

    # Conexão com o banco de dados (SQLite no exemplo)  
    conexao = sqlite3.connect("C:\Denis\TCC\TechnicalDebtDataset_20200606.db")
    cursor = conexao.cursor()

    # Nome da tabela onde deseja inserir os dados
    tabela = '"{}"'.format(project_name)

    # Abre o arquivo CSV e insere os dados no banco de dados
    with open(arquivo_csv, "r", newline="") as arquivo_csv:
        leitor = csv.DictReader(arquivo_csv)
        for linha in leitor:
            # Linhas do CSV
            coluna1 = linha['file'] 
            coluna2 = linha['class']
            coluna3 = linha['type']
            coluna4 = linha['cbo']
            coluna5 = linha['cboModified']
            coluna6 = linha['fanin']
            coluna7 = linha['fanout']
            coluna8 = linha['wmc']
            coluna9 = linha['dit']
            coluna10 = linha['noc']
            coluna11 = linha['rfc']
            coluna12 = linha['lcom']
            coluna13 = linha['lcom*']
            coluna14 = linha['tcc']
            coluna15 = linha['lcc']
            coluna16 = linha['totalMethodsQty']
            coluna17 = linha['staticMethodsQty']
            coluna18 = linha['publicMethodsQty']
            coluna19 = linha['privateMethodsQty']
            coluna20 = linha['protectedMethodsQty']
            coluna21 = linha['defaultMethodsQty']
            coluna22 = linha['visibleMethodsQty']
            coluna23 = linha['abstractMethodsQty']
            coluna24 = linha['finalMethodsQty']
            coluna25 = linha['synchronizedMethodsQty']
            coluna26 = linha['totalFieldsQty']
            coluna27 = linha['staticFieldsQty']
            coluna28 = linha['publicFieldsQty']
            coluna29 = linha['privateFieldsQty']
            coluna30 = linha['protectedFieldsQty']
            coluna31 = linha['defaultFieldsQty']
            coluna32 = linha['finalFieldsQty']
            coluna33 = linha['synchronizedFieldsQty']
            coluna34 = linha['nosi']
            coluna35 = linha['loc']
            coluna36 = linha['returnQty']
            coluna37 = linha['loopQty']
            coluna38 = linha['comparisonsQty']
            coluna39 = linha['tryCatchQty']
            coluna40 = linha['parenthesizedExpsQty']
            coluna41 = linha['stringLiteralsQty']
            coluna42 = linha['numbersQty']
            coluna43 = linha['assignmentsQty']
            coluna44 = linha['mathOperationsQty']
            coluna45 = linha['variablesQty']    
            coluna46 = linha['maxNestedBlocksQty']
            coluna47 = linha['anonymousClassesQty']
            coluna48 = linha['innerClassesQty']
            coluna49 = linha['lambdasQty']
            coluna50 = linha['uniqueWordsQty']
            coluna51 = linha['modifiers']
            coluna52 = linha['logStatementsQty']
            cursor.execute(f"INSERT INTO {tabela} (file, class, type, cbo, cboModified, fanin, fanout, wmc, dit, noc, rfc, lcom, lcomE, tcc, lcc, totalMethodsQty, staticMethodsQty, publicMethodsQty, privateMethodsQty, protectedMethodsQty, defaultMethodsQty, visibleMethodsQty, abstractMethodsQty, finalMethodsQty, synchronizedMethodsQty, totalFieldsQty, staticFieldsQty, publicFieldsQty, privateFieldsQty, protectedFieldsQty, defaultFieldsQty, finalFieldsQty, synchronizedFieldsQty, nosi, loc, returnQty, loopQty, comparisonsQty, tryCatchQty, parenthesizedExpsQty, stringLiteralsQty, numbersQty, assignmentsQty, mathOperationsQty, variablesQty, maxNestedBlocksQty, anonymousClassesQty, innerClassesQty, lambdasQty, uniqueWordsQty, modifiers, logStatementsQty, HASH) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (coluna1, coluna2, coluna3, coluna4, coluna5, coluna6, coluna7, coluna8, coluna9, coluna10, coluna11, coluna12, coluna13, coluna14, coluna15, coluna16, coluna17, coluna18, coluna19, coluna20, coluna21, coluna22, coluna23, coluna24, coluna25, coluna26, coluna27, coluna28, coluna29, coluna30, coluna31, coluna32, coluna33, coluna34, coluna35, coluna36, coluna37, coluna38, coluna39, coluna40, coluna41, coluna42, coluna43, coluna44, coluna45, coluna46, coluna47, coluna48, coluna49, coluna50, coluna51, coluna52, hashCode))

    # Commit para salvar as alterações no banco de dados
    conexao.commit()

    # Feche a conexão com o banco de dados
    conexao.close() 

    print("inserido no banco")

def ComandCmdCk(comand, hash, path):
    #Entramos na pasta do ck
    os.chdir("C:\\Denis\\TCC\ck\\target")
    
    # Executa o comando e captura a saída
    saida = subprocess.check_output(comand, shell=True, text=True)

    # Exibe a saída do comando  
    print(saida)

    #insereBD
    InsertBD(hash, path)  
     

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
        #Executa o CK
        #ComandCmdCk(comando + destinationck_path, commit_hash)
        print(project_name + ' Commit ' + str(contador))   
        ComandCmdCk(comando + destinationck_path, commit_hash, destinationck_path)
    
                
    # Fechar o repositório após a conclusão 
    repo.close()


# Conectar ao banco de dados ou criar um novo arquivo de banco de dados
project_name = "commons-exec"
repo_url = "https://github.com/apache/commons-exec"
local_dir = "C:\Denis\TCC\Projetos\\" + project_name
PercorreCommit(repo_url, local_dir)
   