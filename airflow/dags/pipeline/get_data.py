import os.path
import pandas as pd
import requests
import time
import dotenv
dotenv.load_dotenv()
import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")
URL = os.environ.get("FLASK_URL")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_data():
    url = f"{URL}/get_data"

    response = requests.post(url, proxies={"http": None, "https": None})
    data = response.json()
    df = pd.DataFrame(data=data["data"], columns=data["columns"])

    return df


def update_traffic(a_traffic=0.5, data_size=5000, drift_cols=[]):

    url = f"{URL}/traffic"
    b_traffic = 1 - a_traffic
    payload = {
        "a": round(a_traffic, 2),
        "b": round(b_traffic, 2),
        "data_size": data_size,
        "drift_cols": drift_cols
    }
    response = requests.post(url, json=payload, proxies={"http": None, "https": None})
    print(response.text)


def get_ab_data():
    url = f"{URL}/ab_data"

    response = requests.post(url, proxies={"http": None, "https": None})


    a_data = response.json()["a_data"]
    a_df = pd.DataFrame(data=a_data["data"], columns=a_data["columns"])
    b_data = response.json()["b_data"]
    b_df = pd.DataFrame(data=b_data["data"], columns=b_data["columns"])

    return a_df, b_df


def get_current_split():
    resp = requests.get(url=f"{URL}/traffic")
    return resp


def get_model_version(stage="Production"):
    try:
        latest_model = client.get_latest_versions(name=EXPERIMENT_NAME, stages=[stage])
        version = latest_model[0].version
        run_id = latest_model[0].run_id

        return version, run_id
    except Exception as e:
        print(e)
        print(f"{stage} models not found.")

        return False, False


def get_production_data():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100
    )
    last_run_parent_id = runs["tags.mlflow.parentRunId"].unique()[0]

    mlflow.artifacts.download_artifacts(
        run_id=last_run_parent_id,
        artifact_path="Train.csv",
        dst_path="./temp_dir"
    )


