import pandas as pd
from data_wrangling import *
from plot_utils import salvar_plot

def treinar_modelo(X_train, y_train, biblioteca, model, **config):
    #Importando as bibliotecas
    import importlib
    import numpy as np
    modulo = importlib.import_module(biblioteca)

    #Importando o modelo
    modelo_cls = getattr(modulo, model)

    #Configurando o modelo
    modelo = modelo_cls(**config)

    #Convertendo o y_train para um array 1D
    y_train = np.ravel(y_train)

    #Treinado o modelo
    modelo.fit(X_train, y_train)

    #Mensagem de conclusão
    print(f"\nModelo: {modelo.__class__.__name__}, treinado com sucesso!")
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    #Importando as bibliotecas
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report, roc_curve, auc)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    #Garantindo que o y_test seja um array 1D
    y_test = np.ravel(y_test)

    #Fazendo as previsões
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    #Métricas numéricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #Avaliação do modelo
    print("\n📊 Avaliação do Modelo:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # ==========================
    # 📈 Gráficos e Visualizações
    # ==========================

    # 1️⃣ Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pago", "Não Pago"],
                yticklabels=["Pago", "Não Pago"])
    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"Matriz de Confusão - {modelo.__class__.__name__}")
    salvar_plot(modelo.__class__.__name__, model=True, metrica="Matriz_Confusao")
    plt.tight_layout()
    plt.close()

    # 2️⃣ Curva ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"Curva ROC - {modelo.__class__.__name__}")
        plt.xlabel("Falsos Positivos")
        plt.ylabel("Verdadeiros Positivos")
        plt.legend()
        salvar_plot(modelo.__class__.__name__, model=True, metrica="Curva_ROC")
        plt.close()

    # 3️⃣ Importância das features
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        features = X_test.columns
        importancia = pd.Series(importances, index=features).sort_values(ascending=False)

        # Mostrando o top 10  no console
        print("\nTop 10 variáveis mais importantes:")
        print(importancia.head(10).round(4).to_string())

        #Salvando em CSV
        import os
        import pandas as pd
        pasta_modelo = os.path.join("plots", "models_plots", modelo.__class__.__name__)
        os.makedirs(pasta_modelo, exist_ok=True)
        caminho_csv = os.path.join(pasta_modelo, "Importancia_Features.csv")
        importancia.to_csv(caminho_csv, header=["Importância"], index_label="Variável")
        print(f"📁 Importâncias salvas em: {caminho_csv}\n")

        # 🔹 Gráfico de barras horizontais
        top_n = 8
        top_features = importancia.head(top_n)

        # Para exibir a mais importante no topo, invertemos a ordem para o barh
        top_features_rev = top_features[::-1]

        values = top_features_rev.values
        names = top_features_rev.index

        import matplotlib as mpl

        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')

        plt.gca().set_facecolor('#111111')
        plt.gcf().set_facecolor('#111111')

        # Gradiente (cmap) com normalização pelos valores
        cmap = plt.cm.YlGnBu
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        colors = cmap(norm(values))

        y_pos = np.arange(len(top_features_rev))

        bars = plt.barh(y_pos, values, color=colors, edgecolor='black')

        # Monta rótulos com ranking: o item mais importante tem rank 1
        # Como top_features_rev está invertido, o rank é (top_n - i)
        y_labels = [f"{top_n - i} - {name}" for i, name in enumerate(names)]
        plt.yticks(y_pos, y_labels, fontsize=10)

        plt.xlabel("Importância (proporcional)", fontsize=12)
        plt.title(f"Top {top_n} Features - {modelo.__class__.__name__}", fontsize=14)

        # Ajusta limite x para dar espaço aos textos de porcentagem
        xmax = values.max() * 1.12
        plt.xlim(0, xmax)

        # Adiciona valores em porcentagem ao lado direito das barras
        for i, v in enumerate(values):
            pct = v * 100  # converte para percentuais
            plt.text(v + xmax * 0.01, i, f"{pct:.1f}%", va='center', fontsize=12)

        plt.tight_layout()
        salvar_plot(modelo.__class__.__name__, model=True, metrica="Importancia_Features")
        plt.close()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}

def comparar_modelos(X_train, X_test, y_train, y_test, modelos_dict):
    resultados = {}

    for nome, (biblioteca, classe, config) in modelos_dict.items():
        print(f"\n Treinando modelo: {nome}")
        modelo = treinar_modelo(X_train, y_train, classe, **config)
def main():
    #Lista com os datasets que serão utilizados
    datasets = ["X_train", "X_test", "y_train", "y_test"]

    #Importando os datasets
    X_train, X_test, y_train, y_test = importando_treino_teste(datasets)

    #Separando as colunas em numéricas ou categóricas
    numericas, categoricas = identificar_colunas(X_train)

    #Dicionário com o metodo de tratamento das colunas
    metodos_outliers = {
        'loan_int_rate': 'capping',
        'loan_amnt': 'log',
        'person_income': 'log'
    }

    #Tratamento dos Outliers
    for coluna ,metodo in metodos_outliers.items():
        X_train, X_test = tratar_outliers(X_train, X_test, coluna, metodo)

    #Escalonamento dos valores numéricos
    X_train, X_test = escalar_valores(X_train, X_test, numericas)

    #Transformação das colunas categóricas com OneHotEncoder ou LabelEncoder
    X_train_ready, X_test_ready, todas_cols = transformar_colunas_categoricas(X_train, X_test, numericas, categoricas)

    #Transformando o X_train e o X_test em Dataframes novamente
    X_train_ready = pd.DataFrame(X_train_ready, columns=todas_cols)
    X_test_ready = pd.DataFrame(X_test_ready, columns=todas_cols)

    modelo = treinar_modelo(X_train_ready, y_train, biblioteca="sklearn.ensemble", model="RandomForestClassifier",
                            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    resultados = avaliar_modelo(modelo, X_test_ready, y_test)
if __name__ == "__main__":
    main()