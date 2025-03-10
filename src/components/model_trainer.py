import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score

import mlflow

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            
            mlflow.set_tracking_uri('http://127.0.0.1:5000')
            mlflow.set_experiment(experiment_id=654000327895154401)
            print('MLflow: http://127.0.0.1:5000')
            
            logging.info('MLflow configurado')

            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            logging.info('Divisão das variáveis features e target')

            ## Para treinar o modelo, utilize as células comentadas abaixo
            # with mlflow.start_run():

            #     mlflow.sklearn.autolog()

            #     lr = LogisticRegression(penalty='l1', 
            #                             solver='saga',
            #                             max_iter=1000)

            #     grid_lr = GridSearchCV(lr, param_grid=param_C,
            #                     scoring='roc_auc',
            #                     n_jobs=None,
            #                     refit=True,
            #                     cv=5,
            #                     pre_dispatch='None',
            #                     error_score=np.nan,
            #                     return_train_score=True)

            #     # Pipeline
            #     pipeline = Pipeline([
            #             ('OneHotEnconder', encoder),
            #             ('Scaler', min_max_sc),
            #             ('Model', grid_lr)
            #             ])

            #     pipeline.fit(X_train, y_train)

            #     y_train_predict_proba = pipeline.predict_proba(X_train)
            #     y_test_predict_proba = pipeline.predict_proba(X_test)

            #     train_roc_auc = roc_auc_score(y_train, y_train_predict_proba[:,1])
            #     test_roc_auc = roc_auc_score(y_test, y_test_predict_proba[:,1])

            #     mlflow.log_metrics({'roc_auc_train': train_roc_auc, 'roc_auc_test': test_roc_auc})

            logging.info('Modelo testado e salvo no MLflow')

            client = mlflow.client.MlflowClient()
            version = max([int(i.version) for i in client.get_latest_versions('CreditRisk')])

            if not version:
                raise CustomException('Nenhuma versão de modelo salva, escolha um no MLflow')

            logging.info('Modelo escolhido')
            model = mlflow.sklearn.load_model(f'models:/CreditRisk/{version}')

            return f'Versão do modelo: {version}'

        except Exception as e:
            raise CustomException(e, sys)
