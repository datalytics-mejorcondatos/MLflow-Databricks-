import os
import warnings
import sys

import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
from urllib.parse import urlparse
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


### RandomForest devuelve 0,1 por tanto, creamos un modificador al rededor de la funcion para devolver la probabilidad

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    high_quality = (data.quality >= 7).astype(int)
    data.quality = high_quality
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    n_estimators_ = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',
        'seed': 123}

    def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.xgboost.autolog()
        with mlflow.start_run(nested=True):
            train = xgb.DMatrix(data=train_x, label=train_y)
            test = xgb.DMatrix(data=test_x, label=test_y)
            # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
            # is no longer improving.
            mlflow.log_param('n_estimators_', n_estimators_)
            booster = xgb.train(params=params, dtrain=train, num_boost_round= n_estimators_ ,\
                                evals=[(test, "test")], early_stopping_rounds=50)
            predictions_test = booster.predict(test)
            auc_score = roc_auc_score(test_y, predictions_test)
            mlflow.log_metric('auc', auc_score)

            signature = infer_signature(train_x, booster.predict(train))
            mlflow.xgboost.log_model(booster, "model", signature=signature)
            
            # Set the loss to -1*auc_score so fmin maximizes the auc_score
            return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}
    #spark_trials = SparkTrials(parallelism=10)
    spark_trials = Trials(10)
    with mlflow.start_run(run_name='xgboost_models'):
        best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=50,
        trials=spark_trials, 
        rstate=np.random.RandomState(123))