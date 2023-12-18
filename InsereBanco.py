import csv
import sqlite3

# Nome do arquivo CSV
arquivo_csv = "C:\\Projetos\\commons-jexl_ck\\004ea46842fcaef95a7bd2df1071ef35385f0e76class.csv"

# Conexão com o banco de dados (SQLite no exemplo)
conexao = sqlite3.connect("C:\\DB\\TechnicalDebtDataset_20200606.db")
cursor = conexao.cursor()

# Nome da tabela onde deseja inserir os dados
tabela = "CK"
hash = '123456789ABCDEF'

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
                       (coluna1, coluna2, coluna3, coluna4, coluna5, coluna6, coluna7, coluna8, coluna9, coluna10, coluna11, coluna12, coluna13, coluna14, coluna15, coluna16, coluna17, coluna18, coluna19, coluna20, coluna21, coluna22, coluna23, coluna24, coluna25, coluna26, coluna27, coluna28, coluna29, coluna30, coluna31, coluna32, coluna33, coluna34, coluna35, coluna36, coluna37, coluna38, coluna39, coluna40, coluna41, coluna42, coluna43, coluna44, coluna45, coluna46, coluna47, coluna48, coluna49, coluna50, coluna51, coluna52, hash))

# Commit para salvar as alterações no banco de dados
conexao.commit()

# Feche a conexão com o banco de dados
conexao.close()
