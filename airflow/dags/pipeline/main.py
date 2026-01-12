from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
from scipy.stats import ks_2samp
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from pipeline.get_data import get_ab_data, get_data, update_traffic, get_model_version, get_current_split, \
	get_production_data

KS_PVALUE_THRESHOLD = float(os.environ.get("KS_PVALUE_THRESHOLD"))
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")
if os.environ.get("MLFLOW_LOG") == "true":
	MLFLOW_LOG = True
else:
	MLFLOW_LOG = False


@dag(
	dag_id="main_pipeline",
	start_date=datetime(2025, 1, 1),
	schedule="@daily",  # ежедневный запуск dag'a
	catchup=False,
	tags=["main_pipeline"],
)
def main():
	"""
	Запускает DAG
	:return: None
	"""
	@task
	def check_production_version(stage="Production"):
		"""
		Проверяет наличие Production модели в mlflow model registry
		:param stage:
		:return: словарь с версией модели и уникальный id запуска
		"""
		version, run_id = get_model_version(stage=stage)
		if version:
			return {f"{stage.lower()}_version": version, "run_id": run_id}
		else:
			return {}

	@task
	def train_model(data=None):
		"""
		Обучает модель на предоставленных данных через PyCaret и логирует в mlflow
		:param data: pandas.DataFrame, данные для обучения
		:return: None
		"""
		import mlflow.sklearn
		import pandas as pd
		from pycaret.regression import setup
		from pycaret.regression import compare_models
		import time
		import warnings
		warnings.filterwarnings("ignore", module="matplotlib\..*")

		mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
		mlflow.set_experiment(EXPERIMENT_NAME)
		if data is None:
			df = get_data()
		else:
			df = data["data"]

		cat_cols = df.select_dtypes("object").columns.to_list()
		num_cols = df.select_dtypes("number").drop("price", axis=1).columns.to_list()
		target = "price"
		reg = setup(
			data=df,  # dataset
			target=target,  # target name
			session_id=42,  # for reproducibility
			normalize=True,  # scales numeric features
			categorical_features=cat_cols,  # optional, PyCaret can auto-detect
			numeric_features=num_cols,  # optional, PyCaret can auto-detect
			fold_strategy="kfold",  # cv strategy
			fold=5,  # number of folds
			fold_shuffle=True,  # shuffle data
			n_jobs=-1,  # cores usage
			verbose=True,  # show setup info
			log_experiment=MLFLOW_LOG,  # if true log to mlflow
			experiment_name="price_prediction",
			log_plots=MLFLOW_LOG,
			log_data=MLFLOW_LOG,
			experiment_custom_tags={"created": time.time()}
		)

		best_model = compare_models(include=['lr', 'ridge', 'rf'], sort='MAE', n_select=1, errors="ignore")

		if mlflow.active_run():
			mlflow.end_run()

		print("Training has finished, successfully!")

	@task
	def set_model_stage(result):
		"""
		Устанавливает первично обученную модель в качестве Production для начала полного цикла обучения и сравнения
		:param result: словарь, показывающий была ли найдена Production модель
		:return: None
		"""
		import mlflow
		from mlflow.tracking import MlflowClient

		mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # or your URI
		client = MlflowClient()
		experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

		runs = mlflow.search_runs(
			experiment_ids=[experiment.experiment_id],
			order_by=["start_time DESC"],
			max_results=100)
		last_run_parent_id = runs["tags.mlflow.parentRunId"].unique()[0]
		last_run = runs[runs["tags.mlflow.parentRunId"] == last_run_parent_id]
		best_model_id = last_run[["run_id", "metrics.MAE"]].sort_values("metrics.MAE", ascending=True)["run_id"].iloc[0]
		print(best_model_id)

		model_uri = f"runs:/{best_model_id}/model"

		mlflow.register_model(
			model_uri=model_uri,
			name=EXPERIMENT_NAME
		)

		runs = mlflow.search_runs(
			experiment_ids=[experiment.experiment_id],
			order_by=["start_time DESC"],
			max_results=100)

		last_run_parent_id = runs["tags.mlflow.parentRunId"].unique()[0]
		last_run = runs[runs["tags.mlflow.parentRunId"] == last_run_parent_id]
		best_model_id = last_run[["run_id", "metrics.MAE"]].sort_values("metrics.MAE", ascending=True)["run_id"].iloc[0]
		print(best_model_id)

		model_uri = f"runs:/{best_model_id}/model"

		mlflow.register_model(
			model_uri=model_uri,
			name=EXPERIMENT_NAME
		)

		none_version = client.get_latest_versions(
			name=EXPERIMENT_NAME,
			stages=["None"]
		)

		if not result:
			client.transition_model_version_stage(
				name=EXPERIMENT_NAME,
				version=none_version[0].version,
				stage="Production",
				archive_existing_versions=True
			)

			print("Initial production model was set!")

		else:
			client.transition_model_version_stage(
				name=EXPERIMENT_NAME,
				version=none_version[0].version,
				stage="Staging",
				archive_existing_versions=True
			)

			print("New model was set to Staging!")

	@task
	def compare_distributions():
		"""
		Сравнивает распределения старых и новых данных
		Для числовых признаков используется тест Колмогорова-Смирнова для двух выборок
		Для категориальных признаков применяется PSI тест
		:return:
			result: словарь, содержащий информацию о признаках и было ли определено смещение распределения
		"""
		import random

		get_production_data()

		features_list = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', "price"]

		selected_features = random.sample(features_list, 2)
		update_traffic(a_traffic=0.5, data_size=5000, drift_cols=selected_features)
		ref_data_path = "./temp_dir/Train.csv"
		ref_df = pd.read_csv(ref_data_path, index_col=0)
		cur_df = get_data()

		report = {"data": cur_df, "drift_info": {}}

		cat_cols = ref_df.select_dtypes("object").columns
		num_cols = ref_df.select_dtypes("number").columns

		for num_col in num_cols:
			stat, p_value = ks_2samp(ref_df[num_col].dropna(), cur_df[num_col].dropna())
			report["drift_info"][num_col] = [p_value, 1 if p_value < KS_PVALUE_THRESHOLD else 0]

		for cat_col in cat_cols:
			cur = cur_df[cat_col].value_counts(normalize=True)
			ref = ref_df[cat_col].value_counts(normalize=True)
			dist = pd.concat([ref, cur], axis=1, keys=["ref", "cur"]).fillna(0)
			eps = 1e-6
			dist["ref"] += eps
			dist["cur"] += eps
			dist["psi_component"] = ((dist["ref"] - dist["cur"]) * np.log(dist["ref"] / dist["cur"]))

			psi = dist["psi_component"].sum()

			report["drift_info"][cat_col] = [psi, 1 if psi > PSI_THRESHOLD else 0]

		os.remove(ref_data_path)
		print(report)
		return report

	@task.branch
	def train_compare(result):
		"""
		Выбирает дальнейшее действие: обучение изначальной prod модели или сравнение с уже существующей
		:param result: результат поиска prod модели
		:return: выбранный следующий этап
		"""
		print(result)
		if result:
			version = result["production_version"]
			print(f"Current production version {version}")
			print("Next task - data drift check")
			return "compare_distributions"
		else:
			print("No production models found, the training begins")
			return train_initial_model.operator.task_id

	@task.branch
	def check_drift(report):
		"""
		Выбирает дальнейшее действие после сравнения признаков на смещение: обучение новой staging модели или завершение
		цикла, т.к. смещение не было обнаружено
		:param report: словарь, содержащий информацию о результатах тестов на смещение
		:return: выбранный следующий этап
		"""
		drift_info = report["drift_info"]
		drifted_features = []
		for col, values in drift_info.items():
			drift = values[1]
			if drift == 1:
				drifted_features.append(col)
				print(f"Data drift detected in {col}")

		if drifted_features:
			return train_staging_model.operator.task_id
		else:
			return "check_end"

	@task
	def check_end(decision=None):
		"""
		По результатам сравнения prod и staging модели выводит результат сравнения
		:param decision: результат сравнения
		:return:
		"""
		if decision == "better":
			print("After A/B test new model B was better and was being set as Production model! ")
		elif decision == "worse":
			print("New model has the same or worse performance, keeping current production model.")
		else:
			print("Data drift wasn't detected, keeping current production model.")

	@task
	def ab_test():
		"""
		Проводится A/B тест и логирование в mlflow для определения является ли новая модель лучше, чем текущая
		:return: результат сравнения worse или better
		"""
		import mlflow
		from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

		traffic_info = get_current_split()
		print(traffic_info)

		prod_ver, prod_id = get_model_version(stage="Production")
		stg_ver, stg_id = get_model_version(stage="Staging")

		prod_uri = f'runs:/{prod_id}/model'
		stg_uri = f'runs:/{stg_id}/model'

		a_df, b_df = get_ab_data()

		a_true = a_df.pop("price")
		b_true = b_df.pop("price")

		prod_model = mlflow.pyfunc.load_model(prod_uri)
		stg_model = mlflow.pyfunc.load_model(stg_uri)

		a_pred = prod_model.predict(pd.DataFrame(a_df))
		b_pred = stg_model.predict(pd.DataFrame(b_df))

		a_df["pred_price"] = a_pred
		a_df["price"] = a_true

		b_df["pred_price"] = b_pred
		b_df["price"] = b_true

		a_mae = mean_absolute_error(a_true, a_pred)
		a_rmse = root_mean_squared_error(a_true, a_pred)
		a_r2 = r2_score(a_true, a_pred)

		b_mae = mean_absolute_error(b_true, b_pred)
		b_rmse = root_mean_squared_error(b_true, b_pred)
		b_r2 = r2_score(b_true, b_pred)

		mlflow.set_experiment(os.environ.get("COMPARISON_NAME"))

		if b_mae < a_mae:
			decision = "better"
		else:
			decision = "worse"

		with mlflow.start_run(run_name=f"prod_v{prod_ver}_vs_stg_v{stg_ver}"):
			mlflow.log_param("traffic_info", traffic_info.text)
			mlflow.log_param("prod_model_version", prod_ver)
			mlflow.log_param("prod_model_id", prod_id)
			mlflow.log_param("stg_model_version", stg_ver)
			mlflow.log_param("stg_model_id", stg_id)
			mlflow.log_param("new_prod_model", decision)

			mlflow.log_metric("a_mae", a_mae)
			mlflow.log_metric("a_rmse", a_rmse)
			mlflow.log_metric("a_r2", a_r2)
			mlflow.log_metric("b_mae", b_mae)
			mlflow.log_metric("b_rmse", b_rmse)
			mlflow.log_metric("b_r2", b_r2)

			mlflow.log_table(artifact_file="a_data.json", data=a_df)
			mlflow.log_table(artifact_file="b_data.json", data=b_df)

		return decision

	@task
	def set_new_prod_model():
		"""
		Если новая модель оказывается лучше, чем текущая prod, то она замещает текущую и переносит ее в архив
		:return: None
		"""
		from mlflow import MlflowClient
		import mlflow

		mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
		client = MlflowClient()

		model_ver, model_id = get_model_version("Staging")

		client.transition_model_version_stage(
			name=EXPERIMENT_NAME,
			version=model_ver,
			stage="Production",
			archive_existing_versions=True
		)

	@task.branch
	def model_to_prod(decision):
		"""
		Принимает решение о переводе новой модели в prod
		:param decision: результат сравнения двух моделей
		:return:
		"""
		if decision == "better":
			return new_prod_model.operator.task_id
		else:
			return end_task.operator.task_id

	result = check_production_version(stage="Production")
	train_compare_branch = train_compare(result)
	train_initial_model = train_model.override(task_id="train_initial_model")()
	set_initial_model = set_model_stage.override(task_id="set_initial_model")(result)
	drift_report = compare_distributions()
	train_staging_model = train_model.override(task_id="train_new_model")(drift_report)
	check_drift_branch = check_drift(drift_report)
	staging = set_model_stage.override(task_id="set_staging_model")(result)
	decision = ab_test()
	set_new_model = model_to_prod(decision)
	end_task = check_end.override(task_id="end_model_setup")(decision)

	new_prod_model = set_new_prod_model()

	train_compare_branch >> drift_report >> check_drift_branch >> check_end()
	train_compare_branch >> train_initial_model >> set_initial_model

	check_drift_branch >> train_staging_model >> staging >> decision >> set_new_model

	set_new_model >> end_task
	set_new_model >> new_prod_model


main()
