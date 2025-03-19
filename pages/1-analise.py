import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.utils import DataWrangling

# ----- FUNCTIONS -----

@st.cache_data
def load_data():
    '''
    Carrega e processa os dados do arquivo .csv
    '''
    df = pd.read_csv('artifacts/data.csv')
    df = DataWrangling().fit_transform(df)
    
    return df

def plot_bar(data, x, y, color, title, xlabel, ylabel):
    """
    Cria um gráfico de barras utilizando Plotly Express
    """
    fig = px.bar(
        data,
        x = x,
        y = y,
        color = color,
        title = title,
        labels = {x: xlabel, y: ylabel},
        color_discrete_sequence=['#669bbc', '#bc4749']
    )

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = xlabel,
        yaxis_title = ylabel,
        showlegend=False
    )

    return fig

# ----- DATA -----

df = load_data()

# ----- PAGE -----

st.title('''
    💳 Modelo de Risco de Crédito
    ---
    ''')

st.header('📌 Visão Geral')
    
st.markdown('Esse projeto teve como objetivo identificar potenciais clientes inadimplentes de uma instituição de cartão de crédito. Foram utilizadas técnicas de análise de dados e machine learning para detectar possíveis inadimplências e reduzir prejuízos futuros.')

st.header('💼 Entendimento do Negócio')

st.markdown('''
    De acordo com o Instituto Locomotiva e MFM Tecnologia, oito em cada dez famílias brasileiras estiveram endividadas, e um terço teve dívidas em atraso. Os índices, que haviam piorado significativamente durante a pandemia da covid-19, já recuaram, mas ainda são elevados, segundo o relatório.

    Um dos principais motivos para a inadimplência foi o cartão de crédito, de acordo com a pesquisa. O meio de pagamento foi a fonte de 60% dos débitos em aberto no ano de 2023. Deixar de liquidar dívidas junto a bancos e financeiras, assim como empréstimos e financiamentos, também tem sido um desafio para grande parte dos brasileiros. Uma parcela de 43% lidou com isso atualmente, proporção que subiu em relação ao ano passado, quando era de 40%.

    Essa situação foi prejudicial tanto para os consumidores quanto para as instituições financeiras. Detectar os padrões de consumidores que ficaram inadimplentes nos últimos meses e ter se planejado quanto a isso poderiam ter economizado milhões de reais.
    ---
    ''')
    
st.header('1. Introdução')

st.markdown('''
    O cliente, uma empresa de cartão de crédito, nos trouxe um dataset que incluiu os dados demográficos e financeiros recentes de uma amostra de 30.000 clientes. Esses dados estiveram no nível de conta de crédito (ou seja, uma linha para cada conta). As linhas foram rotuladas de acordo com se, no mês seguinte ao período de dados históricos de seis meses, um proprietário de conta ficou inadimplente, ou seja, não fez o pagamento mínimo.

    **Objetivo**: Nosso objetivo como Cientista de Dados foi desenvolver um modelo, com os dados fornecidos, que previsse se uma conta ficaria inadimplente no próximo mês.
''')


st.header('2. Os dados')
st.subheader('Dicionário')

st.markdown(
    '''
    | Feature                      | Descrição |
    | ---                          | --- |
    | `ID`                         | Coluna de identificação exclusiva da conta |
    | `LIMIT_BAL`                  | valor do crédito fornecido inclusive o crédito do consumidor individual e familiar (complementar) |
    | `SEX`                        | Gênero (1 = masculino; 2 = feminino) |
    | `EDUCATION`                  | Instrução civil (1 = pós-graduação; 2 = universidade; 3 = ensino médio; 4 = outros). |
    | `MARRIAGE`                   | Estado civil (1 = casado; 2 = solteiro; 3 = outros). |
    | `AGE`                        | Idade (ano). |
    | `PAY_1`-`PAY_6`              | Registro do ultimo pagamento. A escala de medida do status de reembolso é a seguinte: -2 = conta começou o mês sem valor a ser pago e o crédito não foi usado; -1 = pagamento pontual; 0 = o pagamento mínimo foi feito, mas o saldo total devedor não foi pago; 1 a 8 = atraso de um a oito mêses no pagamento; 9 = atraso de nove meses ou mais no pagamento. |
    | `BILL_AMT1`-`BILL_AMT6`      | Valor da fatura; BILL_AMT1 representa o valor da fatura em setembro; BILL_AMT2 representa o valor da fatura em agosto; e assim por diante até BILL_AMT7, que representa o valor da fatura em abril. |
    | `PAY_AMT1`-`PAY_AMT6`        | Valor de pagamentos anteriores; PAY_AMT1 representa o valor pago em setembro; PAY_AMT2 representa o valor pago em agosto; e assim por diante até PAY_AMT6, que representa o valor pago em abril. |
    | `default payment next month` | Inadimplência (Alvo) |

    **O dataset conta com:**
    
    - 30000 registros e 25 colunas
    '''
)

st.subheader('Qualidade dos dados')

st.markdown('- Não há valores nulos ou duplicados')

st.subheader('Limpeza e manipulação')

st.markdown('''
    - Foram identificados e removidos registros com valores zerados em todas as suas features;
    - O valor "Not available" da feature `PAY_1` foi alterado para seu valor mais frequente;
    - Valores não identificados no dicionário nas features `EDUCATION` e `MARRIAGE` foram alterados para outros;
    - Uma nova feature de categoria para `EDUCATION` foi criada;
    - Problemas na captura de dados das colunas `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5` e `PAY_6` foram identificados, e por isso, essas colunas foram removidas.
    '''
)

st.header('3. Análise Exploratória dos Dados')
st.subheader('`default payment next month`')

target = df['default payment next month'].value_counts(normalize = True).reset_index()
target['proportion'] = (target['proportion'] * 100).round(2)
target['default payment next month'] = target['default payment next month'].replace({0:'Não', 1:'Sim'})

st.plotly_chart(
    plot_bar(
        target,
        x='default payment next month',
        y='proportion',
        color='default payment next month',               
        title='Distribuição de Inadimplência',
        xlabel = 'Inadimplência',
        ylabel = 'Proporção'),

    use_container_width=True
)

st.markdown(
    '''
    Uma característica muito comum em problemas de inadimplência foi o desbalanceamento de dados. Em nosso dataset, pôde-se observar que apenas 22% da nossa base foi considerada inadimplente.
    '''
)

st.subheader('`PAY_1`')

pay1 = df.groupby('PAY_1')['default payment next month'].mean()
pay1 = (pay1*100).round(2)
fig = px.line(pay1, 
              x=pay1.index,             
              y='default payment next month', 
              markers=True,
              labels={'PAY_1': 'PAY_1', 'default payment next month': 'Proporção de inadimplência'},
              title='Proporção de Inadimplência por PAY_1'
              )

target_mean = df['default payment next month'].mean()
target_mean = (target_mean*100).round(2)

fig.add_trace(go.Scatter(x=df['PAY_1'], 
                         y=[target_mean] * len(df),  
                         mode='lines',
                         line=dict(color='red'),
                         name='Média de inadimplência')
                         )
fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",  # Alinha a parte inferior da legenda
            y=1.14,
            xanchor="left",  # Centraliza a legenda
            x=-0.09
            )
        )       

st.plotly_chart(fig)

st.markdown(
    '''
    O gráfico acima mostrou uma informação muito importante: quem já havia inadimplido apresentou uma tendência maior de fazê-lo novamente. A taxa de inadimplência de contas que estavam em boa situação ficou bem abaixo da taxa geral, e pelo menos 30% das contas que estavam inadimplentes no último mês ficaram inadimplentes novamente.
    '''
)

st.subheader('`LIMIT_BAL`')

limit_bal = df.copy()
limit_bal["default payment next month"] = limit_bal["default payment next month"].replace({0: "Adimplente", 1: "Inadimplente"})

fig = px.histogram(limit_bal, 
                   x='LIMIT_BAL',
                   color="default payment next month",    
                   histnorm='percent',
                   barmode='overlay',
                   opacity=0.4,
                   color_discrete_map={'Adimplente': '#669bbc', 'Inadimplente': '#bc4749'},
                   title=f'Distribuição de LIMIT_BAL por categoria de Inadimplência',
                   nbins=100)
    
fig.update_layout(
        xaxis_title='LIMIT_BAL',
        yaxis_title="Proporção de inadimplência",
        legend=dict(
            title='',
            orientation="h",
            yanchor="top",
            y=1.14,
            xanchor="left",
            x=-0.09
            )
    )

st.plotly_chart(fig)

st.markdown('''
    Aparentemente, contas com limites menores de crédito, de aproximadamente R$ 26.5500, foram relativamente mais propensas a inadimplir. O que fez sentido ao entender que as instituições deram limites menores a contas que apresentavam mais risco de inadimplência.
    ''')

st.subheader('`PAY_AMT`')

# Features de PAY_AMT
pay_amt_feats = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
zero_mask = df[pay_amt_feats] != 0  

# Aplicando log10
pay_amt_log = df[pay_amt_feats][zero_mask].apply(np.log10)
pay_amt = pd.concat([pay_amt_log, df['default payment next month']], axis=1)

# Criando subplots com Plotly
fig = sp.make_subplots(rows=2, cols=3, subplot_titles=pay_amt_feats)

# Definição de cores
colors = {0: '#669bbc', 1: '#bc4749'}

# Loop para adicionar histogramas a cada subplot
for i, col in enumerate(pay_amt_feats):
    row = i // 3 + 1
    col_pos = i % 3 + 1 
    
    for label in [0, 1]:  # 0 = Adimplente, 1 = Inadimplente
        fig.add_trace(
            px.histogram(pay_amt[pay_amt['default payment next month'] == label], x=col, nbins=100, opacity=0.6).data[0],
            row=row, col=col_pos
        )
        fig.data[-1].marker.color = colors[label]
        fig.data[-1].name = "Inadimplente" if label == 1 else "Adimplente"

# Ajustando layout
fig.update_layout(
    title="Distribuição dos Pagamentos (log10)",
    barmode='overlay',
    showlegend=True,
    height=600,
    width=1000
)

st.plotly_chart(fig)

st.markdown('Os gráficos mostraram uma relação entre as variáveis `PAY_AMT` e a variável alvo `default payment next month`. Observou-se que, ao longo do tempo, essa relação foi se tornando menos evidente. Nos pagamentos mais recentes, houve uma distorção na curva para a esquerda, indicando que clientes que realizaram pagamentos menores tiveram uma maior probabilidade de inadimplência.')