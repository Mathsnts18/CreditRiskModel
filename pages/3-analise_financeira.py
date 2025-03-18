import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.utils import DataWrangling
from src.utils import load_object

# ----- FUNCTIONS -----

@st.cache_data
def load_datas():
    '''
    Carrega e processa os dados do arquivo .csv
    '''
    columns = ['LIMIT_BAL', 'PAY_1', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'EDUCATION_graduate school', 'EDUCATION_others', 'default payment next month']

    train_df = pd.read_csv('artifacts/train_processed.csv', names=columns)
    test_df = pd.read_csv('artifacts/test_processed.csv', names=columns)

    df = pd.read_csv('artifacts/data.csv')
    df = DataWrangling().fit_transform(df)

    return train_df, test_df, df

# ----- DATA -----

train_df, test_df, df = load_datas()

# ----- ML -----

model_path='artifacts/model.pkl'
model = load_object(file_path=model_path)

# ----- PREPROCESSOR -----

X_train = train_df.drop('default payment next month', axis=1)
y_train = train_df['default payment next month']

X_test = test_df.drop('default payment next month', axis=1)
y_test = test_df['default payment next month']

# ----- FUNCTIONS -----

def calcular_economia_liquida(custo_aconselhamento, efetividade):
    y_test_predict_proba = model.predict_proba(X_test)
    thresholds = np.linspace(0, 1, 101)

    n_pos_pred = np.empty_like(thresholds)
    custo_total_aconselhamento = np.empty_like(thresholds)
    n_true_pos = np.empty_like(thresholds)
    economia_total = np.empty_like(thresholds)

    for i, threshold in enumerate(thresholds):
        pos_pred = y_test_predict_proba[:, 1] > threshold
        n_pos_pred[i] = sum(pos_pred)
        custo_total_aconselhamento[i] = n_pos_pred[i] * custo_aconselhamento
        true_pos = pos_pred & y_test.astype(bool)
        n_true_pos[i] = sum(true_pos)
        economia_total[i] = n_true_pos[i] * economia_por_inadimplencia * (efetividade / 100)

    economia_liquida = economia_total - custo_total_aconselhamento
    return thresholds, economia_liquida


# ----- PAGE -----

st.title('''
         💸 Análise financeira
         ---
         ''')

st.markdown('Para fazer uma analise financeira, vamos supor o caso de que, para as contas de crédito que estejam em alto risco de inadimplência será oferecido um aconselhamento ao titular que custará R$ 1.300,00 para a empresa com uma taxa de sucesso esperada de 70% para que paguem sua dívida a tempo ou façam acordos alternativos. Os possiveis benefícios do aconselhamento bem-sucedido são que o valor da cobrança mensal de uma conta será percebido como economia, se ela fosse ficar inadimplente, mas não ficou como resultado da conversa.')
            
st.markdown('Portanto, vamos calcular os custos e a economia esperada em um intervalo de limites.')

st.markdown('---')

st.markdown('Para calcular a possivel economia alcançada com a não inadimplência, usaremos o valor médio da ultima fatura mensal de todas as contas')

economia_por_inadimplencia = df['BILL_AMT1'].mean()

st.markdown(f'##### Valor médio da última fatura: **R$ {economia_por_inadimplencia.round(2)}**')

col1, col2 = st.columns(2)

with col1:
    custo_aconselhamento = st.slider(label='Custo por aconselhamento',
                                    min_value=0,
                                    max_value=5000,
                                    value=1300,
                                    step=100
                                    )
with col2:
    efetividade = st.slider(label='Efetividade (%)',
                            min_value=0,
                            max_value=100,
                            value=70,
                            step=1
                            )

# ----- SIMULAÇÃO -----

thresholds, economia_liquida = calcular_economia_liquida(custo_aconselhamento=custo_aconselhamento, efetividade=efetividade)

fig = px.line(
    x=thresholds,
    y=economia_liquida,
    labels={"x": "Limite", "y": "Economia líquida (R$)"},
    title="Economia Líquida vs. Limite"
)

fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=np.linspace(0, 1, 11)),
    showlegend=False
)

st.plotly_chart(fig)

max_savings_ix = np.argmax(economia_liquida)
max_threshold = thresholds[max_savings_ix].round(2)
max_economia = economia_liquida[max_savings_ix].round(2)

st.markdown(f'##### O limite que representa a maior economia é: {max_threshold}')
st.markdown(f'##### A maior economia liquida possivel é de R$ {max_economia}')