import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class DataBalancer:
    def __init__(self, file_path, target_column, output_file='balanced_data.csv'):
        self.file_path = file_path
        self.target_column = target_column
        self.output_file = output_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_res = None
        self.y_train_res = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Primeiras linhas do DataFrame:\n{self.df.head()}")
        print(f"Distribuição original das classes: {Counter(self.df[self.target_column])}")

    def apply_smote(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Ajustar o valor de k_neighbors com base na menor classe
        min_samples = min(Counter(self.y_train).values())
        k_neighbors = max(1, min(min_samples - 1, 1))  # Se min_samples for 1, k_neighbors será 1
        
        if min_samples == 1:
            # Se há uma classe com apenas uma amostra, não podemos aplicar SMOTE diretamente.
            # Vamos remover essas amostras temporariamente para aplicar SMOTE.
            print("Há classes com apenas uma amostra, ajustando o SMOTE.")
            one_sample_classes = [cls for cls, count in Counter(self.y_train).items() if count == 1]
            mask = self.y_train.isin(one_sample_classes)
            X_train_temp = self.X_train[~mask]
            y_train_temp = self.y_train[~mask]
            smote = SMOTE(k_neighbors=1, random_state=42)
            X_train_res_temp, y_train_res_temp = smote.fit_resample(X_train_temp, y_train_temp)
            self.X_train_res = pd.concat([pd.DataFrame(X_train_res_temp), self.X_train[mask]], ignore_index=True)
            self.y_train_res = pd.concat([pd.Series(y_train_res_temp), self.y_train[mask]], ignore_index=True)
        else:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"Distribuição após SMOTE: {Counter(self.y_train_res)}")
        
        # Salvar o DataFrame balanceado em um novo arquivo CSV
        balanced_df = pd.concat([pd.DataFrame(self.X_train_res, columns=self.X_train.columns), pd.DataFrame(self.y_train_res, columns=[self.target_column])], axis=1)
        balanced_df.to_csv(self.output_file, index=False)
        print(f"Dados balanceados salvos em: {self.output_file}")

    def plot_class_distribution(self, y, title):
        counter = Counter(y)
        classes = list(counter.keys())
        counts = list(counter.values())
        
        plt.bar(classes, counts)
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Quantidade')
        plt.show()

    def train_model(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train_res, self.y_train_res)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

    def run(self):
        self.load_data()
        self.apply_smote()
        
        # Visualizar distribuição antes e depois do SMOTE
        self.plot_class_distribution(self.y_train, 'Distribuição das Classes Antes do SMOTE')
        self.plot_class_distribution(self.y_train_res, 'Distribuição das Classes Depois do SMOTE')
        
        #self.train_model()
        #self.evaluate_model()


# Uso da classe
file_path = r'C:\\TCC\\Output\\ML\\TD.csv'
target_column = 'TD'
output_file = r'C:\\TCC\\Output\\ML\\TD_BALANCED.csv'
data_balancer = DataBalancer(file_path, target_column, output_file=output_file)
data_balancer.run()
