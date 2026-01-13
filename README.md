Финальный проект по автоматизации машинного обучения, 3 семесрт.

Выполнил студент группы РИМ-240902 Быцюк Никита

# Структура репозитория
```
mlops_pipeline
│ 
│   docker-compose.yml
│   Dockerfile.airflow
│   requirements.txt
│           
├───airflow
│   ├───dags
│   │   └───pipeline
│   │           get_data.py
│   │           main.py
│   │           
│   └───plugins
│               
├───app
│       app.py
│       DiamondData.csv
│       Dockerfile
│       requirements.txt
│       
├───minio
│       create-bucket.sh
│       
└───mlflow
        basic_auth.ini
        Dockerfile
        requirements.txt

```

# Как запустить

1. Скопируйте репозиторий

```
git clone https://github.com/NikerAi/mlops_pipeline.git
```
2. Создайте файл .env, содержащий следующие переменные. Они необходимы для запуска контейнеров.
```
MLFLOW_TRACKING_URI=http://<название сервиса mlflow из docker_compose>:<ваш порт mlflow>
EXPERIMENT_NAME=<название эксперимента для сохранения логов обучения моделей>
COMPARISON_NAME=<название эксперимента для сохранения логов сравнения моделей через А/В тест>
FLASK_URL=http://<название сервиса flask роутера из docker_compose>:<ваш порт flask>

KS_PVALUE_THRESHOLD=<значение p-value для сравнения распределений>
PSI_THRESHOLD=<допустимое значение разности долей для категоиральных признаков>
MLFLOW_LOG=<значение переменной определяет будут ли логироваться эксперименты, true/false>

AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER>
AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD>
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```
3. Выполните команду из директории, в которую был скопирован репозиторий, для создания контейнеров
```
docker compose up -d
```
4. Проверьте готовность сервисов по адресам
```
mlflow  http://localhost:5000/
minio   http://localhost:9001/
airflow http://localhost:8080/
flask   http://localhost:8000/traffic
```

5. Войдите в airflow http://localhost:8080/ и активируйте main_pipeline, после чего запустятся задачи

5.1 При первом запуске Production модель отсутствует, поэтому запустится ветка обучения первичной модели, которая сразу станет основной/prod моделью.

5.2 При последующих запусках новая модель будет проходить сравнение с основной/prod моделью.

6. Данные для обучения и сравнения моделей предоставляются flask-роутером, который имеет следующие эндпоинты:

6.1 /traffic [GET] для получения текущего состояния параметров конфигурации отправления данных;

6.1.1
```
{
	"a": 0.5, доля данных для модели А
	"b": 0.5, доля данных для модели В
	"data_size": 5000 количество строк отправляемых за один раз
	"drift_cols": [] признаки, в которые необходимо внести смещение распределения
}
```

6.2 /get_data возвращает данные для обучения модели в соответствии с конфигурацией параметров

6.3 /ab_data возвращает данные для А/В теста в соответствии с конфигурацией параметров

6.4 /traffic [POST] позволяет изменить конфигурацию параметров. Изменение можно внести либо через python либо через curl 

```
curl -X POST http://localhost:8000/traffic \
  -H "Content-Type: application/json" \
  -d '{
    "a": 0.5,
    "b": 0.5,
    "data_size": 5000,
    "drift_cols": []
  }'

```
