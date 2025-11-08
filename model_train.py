#Importando biblioteca

import numpy as np
import pandas as pd
from data_preprocessing import import_dataset

def importando_treino_teste(datasets):
    """
    Importa os datasets de treino e teste.
    Parâmetros:
    : datasets ⇾ nome dos datasets, o nome é padronizado:
    Retorna:
    : Dataframes com as features e valores alvo separados em treino e teste:
    """
    data = {name: import_dataset(name) for name in datasets}
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    print("Os dataset foram carregados!")
    for name, df in data.items():
        print(f"{name}: {df.shape}")
    return X_train, X_test, y_train, y_test

def identificar_colunas(df):
    """
    Separa as colunas entre numéricas e categóricas.
    Parâmetros:
    : df ⇾ dataset:
    Retorna:
    : Colunas separadas por tipo dos dados:
    """
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = df.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns.tolist()
    print("Colunas numéricas:", numericas)
    print("Colunas categóricas:", categoricas)
    return numericas, categoricas

datasets = ["X_train", "X_test", "y_train", "y_test"]

X_train, X_test, y_train, y_test = importando_treino_teste(datasets)

numericas, categoricas = identificar_colunas(X_train)

def transformar_colunas(X_train, X_test, numericas, categoricas):
    """
    Transforma as variáveis categóricas
    - As colunas em que a ordem importa, como 'loan_grade', utilizam LabelEncoder.
    - As colunas em que os valores não possuem hierarquia utilizam OneHotEncoder.
    Parâmetros:
    : X_train, X_test ⇾ datasets de treino e teste:
    : numericas, categoricas ⇾ separação por tipo dos dados:
    Retorna:
    : Datasets de treino e teste prontos para o uso no modelo:
    """

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.compose import ColumnTransformer

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    label_encoder = LabelEncoder()
    X_train_encoded['loan_grade'] = label_encoder.fit_transform(X_train_encoded['loan_grade'])
    X_test_encoded['loan_grade'] = label_encoder.transform(X_test_encoded['loan_grade'])

    categoricas_oh = [col for col in categoricas if col != 'loan_grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas_oh),
        ],
        remainder='passthrough'
    )

    X_train_ready = preprocessor.fit_transform(X_train_encoded)
    X_test_ready = preprocessor.transform(X_test_encoded)

    encoded_cols = preprocessor.named_transformers_['categorical'].get_feature_names_out(categoricas_oh)
    todas_cols = list(encoded_cols) + numericas + ['loan_grade']

    print("Formato final dos dados de treino:", X_train_ready.shape)
    print("Formato final dos dados de teste:", X_test_ready.shape)
    print("Total de features após encoding:", len(todas_cols))

    return X_train_ready, X_test_ready, todas_cols

X_train_ready, X_test_ready, todas_cols = transformar_colunas(X_train, X_test, numericas, categoricas)

X_train_ready = pd.DataFrame(X_train_ready, columns=todas_cols)
X_test_ready = pd.DataFrame(X_test_ready, columns=todas_cols)