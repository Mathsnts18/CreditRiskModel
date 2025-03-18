from operator import index
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, DataWrangling, OneHotFeatureEncoder




@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    processed_train_file_path = os.path.join('artifacts', 'train_processed.csv')
    processed_test_file_path = os.path.join('artifacts', 'test_processed.csv')

    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Essa função é responsavel pela tranformação dos dados
        '''

        try:
            numerical_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                                 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                                 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                                 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            std_sc = StandardScaler()

            preprocessor = Pipeline([
                                 ('Scaler', std_sc)
                                ])

            return preprocessor

        except Exception as e:
            CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(filepath_or_buffer=train_path)
            test_df = pd.read_csv(filepath_or_buffer=test_path)

            logging.info('Leitura dos dados de treino e teste completos')

            wrangling = DataWrangling()
            train_df = wrangling.fit_transform(X=train_df)
            test_df = wrangling.transform(X=test_df)
            
            logging.info('Limpeza dos dados de treino e teste completos')

            dummies_train = pd.get_dummies(train_df['EDUCATION_CAT'], dtype=int, prefix='EDUCATION')
            train_df = pd.concat([train_df, dummies_train], axis=1)
            train_df = train_df.drop('EDUCATION_CAT', axis=1)

            dummies_test = pd.get_dummies(test_df['EDUCATION_CAT'], dtype=int, prefix='EDUCATION')
            test_df = pd.concat([test_df, dummies_test], axis=1)
            test_df = test_df.drop('EDUCATION_CAT', axis=1)

            logging.info(msg='Obtendo o objeto de preprocessamento')

            target_column_name = 'default payment next month'

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df = train_df.drop([target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop([target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            selector = SelectKBest(score_func=f_classif, k=10)
            input_feature_train_df = selector.fit_transform(X=input_feature_train_df, y=target_feature_train_df)
            input_feature_test_df = selector.transform(X=input_feature_test_df)
            best_features = selector.get_feature_names_out().tolist()
            
            print(f'As melhores features são: {best_features}')

            logging.info(msg='Melhores features preditoras selecionadas')

            logging.info('Aplicando o objeto de preprocessamento nos dataframes de treino e teste')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                     np.array(target_feature_train_df).reshape(-1,1)]

            test_arr = np.c_[input_feature_test_arr,
                     np.array(target_feature_test_df).reshape(-1,1)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            np.savetxt(self.data_transformation_config.processed_train_file_path, train_arr, delimiter=',')
            np.savetxt(self.data_transformation_config.processed_test_file_path, test_arr, delimiter=',')

            logging.info('Objeto de preprocessamento salvo')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            raise CustomException(e, sys)