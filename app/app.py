import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import logging
import time
np.random.seed(seed=42)

app = Flask(__name__)


TRAFFIC_SPLIT = {
    "a": 0.5,
    "b": 0.5,
    "data_size": 5000,
    "drift_cols": []
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ab-test")


@app.before_request
def log_request():
    request.start_time = time.time()

    logger.info(
        "Incoming request | method=%s path=%s remote_addr=%s json=%s",
        request.method,
        request.path,
        request.remote_addr,
        request.get_json(silent=True)
    )


@app.after_request
def log_response(response):
    duration = time.time() - getattr(request, "start_time", time.time())

    if response.is_json:
        body = response.get_json(silent=True)
    else:
        body = "<non-json response>"

    logger.info(
        "Outgoing response | %s %s %s %.3fs body=%s",
        request.method,
        request.path,
        response.status,
        duration,
        body
    )

    return response


def simulate_drift(data, cols):
    cat_col = data[cols].select_dtypes("object")
    for col in cols:
        if col in cat_col:
            data[col] = np.random.choice(data[col].unique(), size=len(data))
        else:
            data[col] = [
                val * coef if np.random.rand() > 0.5 else val
                for val, coef in zip(data[col].values, np.random.uniform(0.1, 2.0, size=len(data)))
            ]
    return data


@app.route("/get_data", methods=["POST"])
def get_data():
    size = TRAFFIC_SPLIT["data_size"]
    drift_cols = TRAFFIC_SPLIT["drift_cols"]
    data = pd.read_csv("./DiamondData.csv").sample(size)
    if drift_cols:
        data = simulate_drift(data, drift_cols)

    return jsonify(data.to_dict(orient="split"))


@app.route("/ab_data", methods=["POST"])
def ab_data():
    size = TRAFFIC_SPLIT["data_size"]
    drift_cols = TRAFFIC_SPLIT["drift_cols"]
    data = pd.read_csv("./DiamondData.csv").sample(size)
    if drift_cols:
        data = simulate_drift(data, drift_cols)
    a_data = data.sample(int(size*TRAFFIC_SPLIT["a"]))
    b_data = data.drop(a_data.index, axis=0
                       )
    return jsonify({
        "a_data": a_data.to_dict(orient="split"),
        "b_data": b_data.to_dict(orient="split")
    })


@app.route("/traffic", methods=["POST"])
def update_traffic():
    global TRAFFIC_SPLIT
    data = request.json

    if data["a"] + data["b"] != 1.0:
        return jsonify({"error": "Traffic split must sum to 1.0"}), 400

    TRAFFIC_SPLIT = data
    return jsonify({"status": "updated", "traffic": TRAFFIC_SPLIT})


@app.route("/traffic", methods=["GET"])
def get_traffic():
    return jsonify(TRAFFIC_SPLIT)


# -----------------------------
app.run(host="0.0.0.0", port=8000)
