#Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)

def salvar_plot(nome: str):
    path = os.path.join('plots', f"{nome}" + '.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print('Gráfico salvo em:', path)

#Importando o dataset
df = pd.read_csv('data/emprestimos_concebidos.csv')

#Describe e info
print(df.describe())
print(df.info())

#Verificando valores nulos
print(df.isnull().sum())
print((df.isnull().mean() * 100).round(2))
"""
Duas colunas possuem valores nulos:
person_emp_length              895
loan_int_rate                 3116
A coluna person_emp_length possui algo próximo de 3% do total como valores nulos,
portanto utilizar o dropna não deve ser tão problematico.
Já a loan_int_rate é bem mais expressiva, possuindo quase 10% do total. 
Portanto optarei por substituir os valores nulos pela mediana utilizando o loan_grade como base para tal.
"""

#Tratando valores nulos
df = df.dropna(subset=['person_emp_length'])

df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
    lambda x: x.fillna(x.median())
)

#Conferindo se sobrou algum valor nulo
print(df.isnull().sum())

#Verificando a porcentagem de empréstimos pagos ou não
print(df['loan_status'].value_counts(normalize=True))

#Analisando a Correlação entre o status com o valor do empréstimo
print(df.groupby('loan_status')['loan_amnt'].mean())

#Analisando a Correlação entre o status e a existência de um empréstimo não pago
print(df.groupby('loan_status')['cb_person_default_on_file'].value_counts(normalize=True))

#Analisando a Correlação entre o status e o juros do empréstimo
print(df.groupby('loan_status')['loan_int_rate'].mean())

#Com essas comparações deu pra notar que o loan_status 0 se refere a prováveis bons pagadores
#enquanto loan_status 1 é para possíveis inadimplentes

#Boxplot para detectar outliers na renda
sns.boxplot(data=df, y='person_income')
plt.title('Distribuição de Renda')
plt.show()
plt.close()
"""
Análise do gráfico:
Os out-liers já começam antes mesmo do primeiro milhão
"""

#Linha para ver melhor a distribuição de renda
sns.kdeplot(data=df["person_income"])
plt.title("Linha kde para renda")
plt.show()
plt.close()
"""
Analise do gráfico:
Continuando com a lógica do gráfico anterior o pico da linha fica próximo ao 0
"""

#Verificando a distribuição do loan_status
df['loan_status_label'] = df['loan_status'].map({0: 'Pago', 1: 'Não pago'})
ax = sns.countplot(x='loan_status_label', data=df, hue='loan_status', order=['Pago', 'Não pago'])
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10)
salvar_plot("Countplot_Status_Emprestimo")
plt.show()
plt.close()
"""
Analise do gráfico:
O número de pessoas que pagaram o empréstimo foi de 25473, já o total de inadimplentes foi de 7108
Com isso é possível notar que 78% das pessoas acabam pagando os empréstimos.
"""

#Verificando a correlação entre as variáveis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True).round(2), annot=True, cmap='coolwarm', fmt='.2f',)
plt.xticks(rotation=45)
salvar_plot("Heatmap")
plt.show()
"""
Análise do gráfico:
A variável com a maior correlação com casos em que o empréstimo não foi pago é:
Percentual da renda que será destinada ao pagamento do empréstimo.
"""

sns.boxplot(data=df, y='loan_status_label', x='loan_percent_income')
plt.title('Distribuição de Pagamento por Porcentagem da Renda')
salvar_plot("Boxplot_Pagamento_Por_Porcentagem_Renda")
plt.show()
"""
Análise do gráfico:
O percentual da renda que é destinado ao pagamento se mostra novamente um fator determinante para o pagamento do empréstimo.
Os empréstimos que foram pagos haviam um percentual menor em relação a renda
"""

#Nova coluna com as rendas divididas por 1000 para uma melhor visualização
df["person_income_k"] = df["person_income"]/1000

#Separando por faixas de renda
bins = [0, 50, 100, 500, 1000, float('inf')]
labels = ["<50k", "50~100k", "100~500k", "500k~1m", ">1m"]
df['income_range'] = pd.cut(df["person_income_k"], bins=bins, labels=labels)

#Gráfico para ver a distribuição do loan_status por faixa de renda
sns.countplot(data=df, x='income_range', hue='loan_status_label')
plt.show()
"""
Analise do gráfico:
Com o gráfico fica fácil notar que a enorme maioria das pessoas que pediram empréstimos
possuem uma renda anual abaixo de 500 mil.
Além disso o maior percentual de não pagadores está na faixa de renda anual menor que 50 mil
"""

#Verificando a o valor total de casos por faixa de renda
print(df['income_range'].value_counts())

#Como existem poucos casos acima de 500k eu decidi juntar as faixas de 500 mil e mais de 1m
df['income_range'] = df['income_range'].astype(str).replace(['500k~1m', '>1m'], '>500k')

#Mesmo gráfico que o anterior, porém feito após a junção das faixas de renda acima de 500 mil
sns.countplot(data=df, x='income_range', hue='loan_status_label', order=["<50k", "50~100k", "100~500k", ">500k"])
salvar_plot("Countplot_Pagamento_Por_Faixa_Renda")
plt.show()

"""
Resumo do EDA:
- 78% dos empréstimos foram pagos integralmente.
- Pessoas com renda anual < 50k apresentam maior taxa de inadimplência.
- A correlação mais forte com inadimplência é a razão 'loan_percent_income'.
- O tratamento de nulos foi realizado de forma consistente, preservando a integridade dos dados.
"""