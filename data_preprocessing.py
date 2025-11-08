#Importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split

#Funções do pré-processamento de dados

#Função para importar datasets na pasta data
def import_dataset(df):
    """
    Importa um arquivo CSV da pasta data/ e retorna um DataFrame.
    Parâmetros:
    : df ⇾ nome do arquivo (sem extensão):
    Retorna:
    : Dataframe com os dados:
    """
    return pd.read_csv(f"./data/{df}.csv")

#Função para tratar os valores nulos, substituindo eles pela mediana da loan_grade do conjunto de treino
def preencher_mediana_treino_teste(X_train, X_test):
    """
    Substitui valores nulos em 'loan_int_rate' pela mediana correspondente ao 'loan_grade',
    calculada apenas no conjunto de treino. Evita data leakage e garante consistência.
    """
    medianas = X_train.groupby('loan_grade')['loan_int_rate'].median()

    for grade, mediana in medianas.items():
        X_train.loc[X_train['loan_grade'] == grade, 'loan_int_rate'] = (
            X_train.loc[X_train['loan_grade'] == grade, 'loan_int_rate'].fillna(mediana)
        )
        X_test.loc[X_test['loan_grade'] == grade, 'loan_int_rate'] = (
            X_test.loc[X_test['loan_grade'] == grade, 'loan_int_rate'].fillna(mediana)
        )

    return X_train, X_test

#Função para Exportar o dataset
def export_dataset(df, name, folder='data'):
    """
    Exporta um arquivo CSV para a pasta "data"
    Parâmetros: df ⇾ dataframe:
    : name ⇾ nome do arquivo:
    : folder ⇾ nome da pasta:
    """
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.csv")
    try:
        df.to_csv(path, index=False)
        print(f"{name}.csv exportado com sucesso!")
    except Exception as e:
        print(f"Erro ao exportar o dataset {name}: {e}")


#Pré-processamento

def main():

    #Importando dataset
    df = import_dataset("emprestimos_concebidos")

    #Já sabemos pela EDA que o dataset possui valores nulos
    print(df.isnull().sum())

    #Dropando os nulos da coluna person_emp_length
    df = df.dropna(subset=["person_emp_length"])

    #Separando o valor alvo do resto
    y = df["loan_status"]
    X = df.drop(columns=["loan_status"])

    #Separando o dataset entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Verificando se deu certo
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    #Analisando quantos valores nulos ficaram em cada lado
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())

    #Aplicando a função acima
    X_train, X_test = preencher_mediana_treino_teste(X_train, X_test)

    #Confirmando se todos os valores nulos foram corrigidos
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())

    #Exportando os datasets
    export_dataset(X_train, name="X_train")
    export_dataset(X_test, name="X_test")
    export_dataset(y_train, name="y_train")
    export_dataset(y_test, name="y_test")

    print("\n✅ Pré-processamento concluído com sucesso!")
    print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")

if __name__ == "__main__":
    main()