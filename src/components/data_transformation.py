import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, DataWrangling, OneHotFeatureEncoder




@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Essa função é responsavel pela tranformação dos dados
        '''

        try:

            encoder = OneHotFeatureEncoder()
            std_sc = StandardScaler()

            preprocessor = Pipeline([
                                 ('OneHotEnconder', encoder),
                                 ('Scaler', std_sc)
                                ])

            return preprocessor

        except Exception as e:
            CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Leitura dos dados de treino e teste completos')

            wrangling = DataWrangling()
            train_df = wrangling.fit_transform(train_df)
            test_df = wrangling.transform(test_df)

            logging.info('Limpeza dos dados de treino e teste completos')

            logging.info('Obtendo o objeto de preprocessamento')

            target_column_name = 'default payment next month'

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

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

            logging.info('Objeto de preprocessamento salvo')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            raise CustomException(e, sys)