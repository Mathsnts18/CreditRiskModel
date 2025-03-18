from csv import Error
import sys
import os
from dataclasses import dataclass
from tracemalloc import stop
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

import mlflow
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Configurando o MLflow')
            
            try:
                mlflow.set_tracking_uri('http://127.0.0.1:5000')
                print('MLflow: http://127.0.0.1:5000')
                id_experimento = input("Digite o ID do experimento: ")
                mlflow.set_experiment(experiment_id=int(id_experimento))
                    
            except Exception:
                raise CustomException('ID inválido.', sys)
                
            
            
            logging.info('MLflow configurado')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            logging.info('Divisão das variáveis features e target')

            ## Para treinar o modelo, utilize as células comentadas abaixo
            # with mlflow.start_run():

            #     mlflow.sklearn.autolog()

            #     model = 

            #     # Pipeline
            #     pipeline = Pipeline([
            #            
            #             ])

            #     pipeline.fit(X_train, y_train)

            #     y_train_predict_proba = pipeline.predict_proba(X_train)
            #     y_test_predict_proba = pipeline.predict_proba(X_test)

            #     train_roc_auc = roc_auc_score(y_train, y_train_predict_proba[:,1])
            #     test_roc_auc = roc_auc_score(y_test, y_test_predict_proba[:,1])

            #     mlflow.log_metrics({'roc_auc_train': train_roc_auc, 'roc_auc_test': test_roc_auc})

            # logging.info('Modelo testado e salvo no MLflow')

            client = mlflow.client.MlflowClient()
            version = max([int(i.version) for i in client.get_latest_versions('CreditRisk')])

            if not version:
                raise CustomException('Nenhuma versão de modelo salva, escolha um no MLflow')

            logging.info('Modelo escolhido')
            model = mlflow.sklearn.load_model(f'models:/CreditRisk/{version}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            return f'Versão do modelo: {version}'

        except Exception as e:
            raise CustomException(e, sys)
