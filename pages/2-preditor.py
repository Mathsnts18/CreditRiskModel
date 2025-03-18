import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.utils import DataWrangling
from src.utils import load_object

# ----- FUNCTIONS -----

@st.cache_data
def load_data():
    '''
    Carrega e processa os dados do arquivo .csv
    '''
    df = pd.read_csv('artifacts/data.csv')
    df = DataWrangling().fit_transform(df)
    
    return df

# ----- DATA -----

df = load_data()

# ----- MODEL -----

model_path='artifacts/model.pkl'
model = load_object(file_path=model_path)

# ----- PAGE -----

st.title('''
         🤖 Preditor de Risco de Inadimplência
         ---
         ''')
st.subheader('Insira os dados e calcule a probabilidade do cliente inadimplir')

# Inputs do usuário

pay1 = st.selectbox('Último pagamento', ['Sem uso de crédito',
                                         'Pago',
                                         'Pagamento mínimo',
                                         'Atraso de 1 mês',
                                         'Atraso de 2 meses',
                                         'Atraso de 3 meses',
                                         'Atraso de 4 meses',
                                         'Atraso de 5 meses',
                                         'Atraso de 6 meses',
                                         'Atraso de 7 meses',
                                         'Atraso de 8 meses',
                                         '9 meses ou mais'])

education = st.selectbox('Formação', ['Pós-graduação',
                                      'Ensino superior',
                                      'Ensino médio',
                                      'Outros'])                                                                                                                                                                                                                                                                                                                                       

limit_bal = st.number_input('Valor de crédito fornecido')

pay_amt1 = st.number_input('Valor pago em setembro')

pay_amt2 = st.number_input('Valor pago em agosto')

pay_amt3 = st.number_input('Valor pago em julho')

pay_amt4 = st.number_input('Valor pago em junho')

pay_amt5 = st.number_input('Valor pago em maio')

pay_amt6 = st.number_input('Valor pago em abril')

# Dicionário de entrada

input_features = {
    'LIMIT_BAL': limit_bal,
    'PAY_1': pay1, 
    'PAY_AMT1': pay_amt1, 
    'PAY_AMT2': pay_amt2, 
    'PAY_AMT3': pay_amt3, 
    'PAY_AMT4': pay_amt4,
    'PAY_AMT5': pay_amt5, 
    'PAY_AMT6': pay_amt6, 
    'EDUCATION': education
}

input_df = pd.DataFrame(input_features, index=[0])

map_pay1 = {
    'Sem uso de crédito': -2,
    'Pago': -1,
    'Pagamento mínimo': 0,
    'Atraso de 1 mês': 1,
    'Atraso de 2 meses': 2,
    'Atraso de 3 meses': 3,
    'Atraso de 4 meses': 4,
    'Atraso de 5 meses': 5,
    'Atraso de 6 meses': 6,
    'Atraso de 7 meses': 7,
    'Atraso de 8 meses': 8,
    '9 meses ou mais': 9
}
map_education = {
    'Pós-graduação': 'graduate school',
    'Ensino superior': 'university',
    'Ensino médio': 'high school',
    'Outros': 'others' 
}

input_df['PAY_1'] = input_df['PAY_1'].map(map_pay1)
input_df['EDUCATION'] = input_df['EDUCATION'].map(map_education)
input_df['EDUCATION_graduate school'] = np.where(input_df['EDUCATION'] == 'graduate school', 1, 0)
input_df['EDUCATION_others'] = np.where(input_df['EDUCATION'] == 'others', 1, 0)
input_df.drop('EDUCATION', axis=1, inplace=True)

with st.container():
    if st.button('Previsão'):
        prob = (model.predict_proba(input_df)[:,1][0])*100
        st.markdown(f'##### O cliente tem {prob.round(2)}% de chance de inadimplir')