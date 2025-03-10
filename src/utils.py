import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class DataWrangling(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Removendo as linhas com apenas 0
        X_zero = X_copy == 0
        feats_zero = X_zero.iloc[:,1:].all(axis=1)
        X_copy = X_copy.loc[~feats_zero,:]

        # Alterando o 'PAY_1' para seu valor mais frequente
        X_copy = X_copy.replace({'PAY_1': 'Not available'}, X_copy['PAY_1'].mode().values[0])
        X_copy['PAY_1'] = X_copy['PAY_1'].astype(int)

        # Alterando valores não identificados para outros
        X_copy = X_copy.replace({'EDUCATION': [0, 5, 6]}, 4)
        X_copy = X_copy.replace({'MARRIAGE': 0}, 3)

        # Criando nova feature
        edu_cat_mapping = {
            1: 'graduate school',
            2: 'university',
            3: 'high school',
            4: 'others'
        }
        X_copy['EDUCATION_CAT'] = X_copy['EDUCATION'].map(edu_cat_mapping)

        # Removendo as features PAY_
        X_copy = X_copy.drop(['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1)

        return X_copy

class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)  # Configurar sparse para False para retornar array denso
        self.cols = ['EDUCATION_CAT']
    
    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self
    
    def transform(self, X):
        onehot_data = self.encoder.transform(X[self.cols])
        
        # Convertendo o array onehot_data em um DataFrame
        onehot_df = pd.DataFrame(onehot_data, columns=self.encoder.get_feature_names_out(self.cols))
        
        # Para garantir que os índices estejam alinhados, vamos redefinir o índice do onehot_df para corresponder ao de X
        onehot_df.index = X.index
        
        X = X.drop(self.cols, axis=1)
        X = pd.concat([X, onehot_df], axis=1)
        
        return X