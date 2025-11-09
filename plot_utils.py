def salvar_plot(nome: str, pasta_base='plots', timestamp=False, model=False, metrica=False):
    """
    Verifica se o gráfico é de um modelo
    Verifica se o Timestamp é True
    Cria subpastas automaticamente caso necessário.
    Salva o gráfico atual (plt) como imagem PNG dentro da pasta definida.
    Exemplo:
        salvar_plot(modelo.__class__.__name__, model=True, metrica="Matriz_Confusao")
    Isso salvará em:
        plots/models_plots/Nome_do_modelo/Matriz_Confusao.png
    """
    #Bibliotecas necessárias
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    #Atribui a pasta do modelo correspondente caso necessário
    if model:
        pasta_base = pasta_base + "/models_plots/" + nome
        nome_arquivo = f"{metrica}.png"
        #Atribui Timestamp se habilitado
        if timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            nome_arquivo = f"{metrica}_{timestamp}.png"
    else:
        #Atribui Timestamp se habilitado
        if timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            nome_arquivo = f"{nome}_{timestamp}.png"
        else:
            nome_arquivo = f"{nome}.png"

    #Atribui o caminho completo
    caminho_completo = os.path.join(pasta_base, nome_arquivo)

    #Garantindo que todas as subpastas existam
    os.makedirs(os.path.dirname(caminho_completo), exist_ok=True)

    #Salvando o gráfico
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {caminho_completo}\n")