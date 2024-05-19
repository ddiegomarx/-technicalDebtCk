
import git

# URL do repositório que você deseja clonar
repo_url = "https://github.com/apache/commons-jexl"

# Diretório local onde o repositório será clonado
local_dir = "C:\Projetos\commons-jexl"

# Clone o repositório
git.Repo.clone_from(repo_url, local_dir)

print("Repositório clonado com sucesso em:", local_dir)
