import sqlite3
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# Conectar ao banco de dados SQLite
conn = sqlite3.connect('C:\TCC\BD\ck\Metricas_ck.db')
cursor = conn.cursor()

# Extrair os dados da tabela existente
cursor.execute('SELECT * FROM SIMPLE_DATASET_NORMATIZADO')
dados = cursor.fetchall()

# Converter os dados em um array numpy
dados_array = np.array(dados)

# Selecionar as colunas que serão normalizadas
coluna_nao_normalizada = dados_array[:, :5]  # Seleciona as quatro primeiras colunas

# Deixar a terceira coluna sem normalização
dados_normalizar = dados_array[:, 5:]  # 5 em diante normalizar

# Lidar com valores ausentes
imputer = SimpleImputer(strategy='mean') 

# Normalizar os dados
colunas_normalizar_sem_nan = imputer.fit_transform(dados_normalizar)
dados_normalizados = preprocessing.normalize(colunas_normalizar_sem_nan)

# Inserir os dados normalizados na nova tabela
for i in range(len(dados_normalizados)):
    valores = tuple(list(coluna_nao_normalizada[i]) + list(dados_normalizados[i]))
    cursor.execute('INSERT INTO TABLE_DATASET_NORMALIZE_V2 (QUALIDADE_CODIGO, TD_ATUAL, TD_NOVA, PROJETO_CODIGO, FILE, "TYPE", sqaleRating, sqaleDebtRatio, newSqaleDebtRatio, reliabilityRemediationEffort, securityRemediationEffort, effortToReachMaintainabilityRatingA, watchCount, RESOLUTION_CODIGO, complexity, duplicatedLines, duplicatedBlocks, duplicatedFiles, codeSmells, bugs, uncoveredLines, vulnerabilities, file_type, cbo, cboModified, fanin, fanout, wmc, dit, noc, rfc, lcom, lcomE, tcc, lcc, totalMethodsQty, staticMethodsQty, publicMethodsQty, privateMethodsQty, protectedMethodsQty, defaultMethodsQty, visibleMethodsQty, abstractMethodsQty, synchronizedMethodsQty, totalFieldsQty, staticFieldsQty, publicFieldsQty, privateFieldsQty, protectedFieldsQty, defaultFieldsQty, finalFieldsQty, nosi, loc, returnQty, loopQty, comparisonsQty, tryCatchQty, parenthesizedExpsQty, stringLiteralsQty, numbersQty, assignmentsQty, mathOperationsQty, variablesQty, maxNestedBlocksQty, anonymousClassesQty, innerClassesQty, lambdasQty, uniqueWordsQty, modifiers, logStatementsQty, ReplaceVariableWithAttribute, InlineVariable, MoveAttribute, InlineMethod, MoveClass, ExtractAndMoveMethod, ExtractClass, RenameClass, RenameParameter, RenameAttribute, RenameMethod, PullUpMethod, RenameVariable, ExtractVariable, ExtractMethod) VALUES (' + ', '.join(['?'] * len(valores)) + ')', valores)

# Commit para salvar as alterações
conn.commit()

# Fechar a conexão
conn.close()
