import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Inicialização do método ou componente de ingestão de dados')
        
        try:
            df = pd.read_excel('data/raw/default_of_credit_card_clients__courseware_version_1_21_19.xls')
            logging.info('Lendo o dataset como um dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Separação de treino e teste iniciada')
            train_set, test_set = train_test_split(df, 
                                                   test_size=0.2,
                                                   stratify=df['default payment next month'],
                                                   random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestão de dados completo!')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()