import streamlit as st

st.set_page_config(
    page_title='CreditRiskModel',
    page_icon='💳'
)

analise_page = st.Page('pages/1-analise.py', title='Análise e Insights', icon='💡')
preditor_page = st.Page('pages/2-preditor.py', title='Preditor', icon='🤖')
financeira_page = st.Page('pages/3-analise_financeira.py', title='Análise financeira', icon='💸')

pg = st.navigation([analise_page, preditor_page, financeira_page])

pg.run()