import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import uri
mlflow.set_tracking_uri("databricks")
mlflow.projects.run("##ProjecsPath##",
                    experiment_id = "##DatabricksExperimentID##",
                    backend = "databricks",
                    backend_config = "cluster-spec.json")