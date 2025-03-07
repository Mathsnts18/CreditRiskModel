from sklearn.base import BaseEstimator, TransformerMixin

class DataProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Alterando o 'PAY_1' para seu valor mais frequente
        X_copy = X_copy.replace({'PAY_1': 'Not available'}, 
                                 X_copy['PAY_1'].mode().values)

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
