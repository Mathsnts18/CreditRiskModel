import streamlit as st
import pandas as pd
import plotly.express as px
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
    
st.markdown('Esse projeto visa identificar potenciais clientes inadimplentes de uma instituição de cartão de crédito. Foram utilizadas técnicas de análise de dados e machine learning para detectar possiveis inadimplências e reduzir prejuízos futuros.')

st.header('💼 Entendimento do Negócio')

st.markdown('''
    De acordo com o Instituto Locomotiva e MFM Tecnologia, oito em cada dez familias brasileiras estão individades e um terço têm dívidas em atraso. Os índices, que haviam piorado significativamente durante a pandemia da covid-19, já recuaram, mas ainda são elevados, segundo o relatório.

    Um dos principais motivos para a inadimplência é o cartão de crédito, de acordo com a pesquisa. O meio de pagamentos foi a fonte de 60% dos débitos em aberto no ano de 2023. Deixar de liquidar dívidas junto a bancos e financeiras e empréstimos e financiamentos também tem sido um desafio para grande parte dos brasileiros. Uma parcela de 43% lida com isso atualmente, proporção que subiu em relação ao ano passado, quando era de 40%.

    Essa situação é prejudicial tanto para os consumidores quanto para as instituições financeiras. Detectar os padrões de consumidores que ficarão inadimplentes nos próximos meses e se planejar quanto a isso podem economizar milhões de reais.

    ---
    ''')
    
st.header('1. Introdução')

st.markdown('''
    O cliente, uma empresa de cartão de crédito, nos trouxe um dataset que inclui os dados demográficos e financeiros recentes de uma amostra de 30.000 clientes. Esses dados estão no nível de conta de crédito (ou seja, uma linha para cada conta). As linhas são rotuladas de acordo com se no mês seguinte ao período de dados histórico de seis meses um proprietário de conta ficou inadimplente, ou seja, não fez o pagamento mínimo.

    **Objetivo**: Nosso objetivo como Cientista de Dados é desenvolver um modelo, com os dados fornecidos, que preveja se uma conta ficará inadimplente no próximo mês.
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
    - Foram identificados e removidos registros com valores zerados em todos suas features;
    - Alterado o valor "Not available" da feature `PAY_1` para seu valor mais frequente;
    - Alterado valores não identificados no dicionário nas features `EDUCATION` e `MARRIAGE` para outros;
    - Criado nova feature de categoria para `EDUCATION`
    - Foram identificados problemas na captura de dados das colunas `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5` e `PAY_6` e por isso foram removidas;
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
    Uma característica muito comum em problemas de inadimplência é o desbalanceamento de dados. Em nosso dataset podemos observar que apenas 22% da nossa base é considerada inadimplente.
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
    O gráfico acima mostra uma informação muito importante: quem já inadimpliu apresenta uma tendência maior de fazê-lo novamente. A taxa de inadimplência de contas em boa situação está bem abaixo da taxa geral e pelo menos 30% das contas que estavam inadimplentes no último mês estarão inadimplentes novamente.
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

st.markdown('Aparentemente, contas com limites menores de créditos, de aproximadamente NT$ 150,000 são relativamente mais propensas a inadimplir. O que faz sentido ao entender que as instituições dão limites menores a conntas que apresentam mais risco de inadimplência.')