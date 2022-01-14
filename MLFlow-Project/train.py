# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

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


# mlflow: Creamos un run que sirve para guiar el experimento. 
# mediante mlflow.log* podemos crear snapshots del modelo y registrarlos 
# mlflow.log_metric podemos guardar las metricas de desesmpeño.

    with mlflow.start_run(run_name = "untuned_random_forest"):
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        n_estimators = n_estimators_
        mlflow.log_param('n_estimators_', n_estimators)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
        model.fit(train_x, train_y)

        predictions_test = model.predict_proba(test_x)[:,1]
        auc_score = roc_auc_score(test_y, predictions_test)
        mlflow.log_metric('auc', auc_score)

        wrappedModel = SklearnModelWrapper(model)
        signature = infer_signature(train_x, wrappedModel.predict(None, train_x))


        conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None)

        # Model registry falla con file store en ocasiones
        if tracking_url_type_store != "file":
            mlflow.pyfunc.log_model("rf_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
        else:
            mlflow.sklearn.log_model(model, "rf_model")
