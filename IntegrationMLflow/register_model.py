import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import uri
mlflow.set_tracking_uri("databricks")
runs = mlflow.search_runs(experiment_ids="##DatabricksExperimentID##")
print(runs.head())
best_id  = runs.sort_values("metrics.auc", ascending=False)["run_id"][0]
artifacts_path = runs[runs.run_id == best_id]["artifact_uri"][0]
print(artifacts_path)

model_name = "pdn_model"
model_version = mlflow.register_model(f"{artifacts_path}/rf_model", model_name)

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)