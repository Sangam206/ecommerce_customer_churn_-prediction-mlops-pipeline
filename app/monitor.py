import pickle
import logging
import pandas as pd

from prometheus_client import Gauge, start_http_server

from evidently import Report, Dataset, DataDefinition
from evidently import BinaryClassification          # for classification mapping
from evidently.presets import DataDriftPreset, ClassificationPreset


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
X_TRAIN_PATH = "after encoding/x_train.csv"
X_TEST_PATH = "after encoding/x_test.csv"
Y_TRAIN_PATH = "splitting data/y_train.csv"
Y_TEST_PATH = "splitting data/y_test.csv"
MODEL_PATH = "models/model.pkl"


# ─────────────────────────────────────────────
# PROMETHEUS / LOGGING
# ─────────────────────────────────────────────
DRIFT_GAUGE = Gauge("data_drift_score", "Data drift score")
ACC_GAUGE = Gauge("model_accuracy", "Model accuracy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_prometheus():
    start_http_server(8000)
    print("Prometheus running → http://localhost:8000/metrics")


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_data():
    logger.info("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_train, X_test, y_train, y_test


def load_model():
    logger.info("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# PREPARE
# ─────────────────────────────────────────────
def prepare(X_train, X_test, y_train, y_test, model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    reference = X_train.copy()
    reference["target"] = y_train.astype(str).values
    reference["prediction"] = pd.Series(train_pred).astype(str).values

    current = X_test.copy()
    current["target"] = y_test.astype(str).values
    current["prediction"] = pd.Series(test_pred).astype(str).values

    return reference, current


# ─────────────────────────────────────────────
# HELPER: pull a value out of the snapshot dict by metric id substring
# ─────────────────────────────────────────────
def extract_metric(result_dict, *id_keywords):
    """
    Walks the snapshot's dict and returns the first metric value
    whose metric_id contains all given keywords (case-insensitive).
    """
    for m in result_dict.get("metrics", []):
        mid = str(m.get("metric_id", "")).lower()
        if all(k.lower() in mid for k in id_keywords):
            val = m.get("value")
            if isinstance(val, dict):
                # some metrics nest the number, grab first numeric leaf
                for v in val.values():
                    if isinstance(v, (int, float)):
                        return float(v)
            elif isinstance(val, (int, float)):
                return float(val)
    return 0.0


# ─────────────────────────────────────────────
# DRIFT  (no target/prediction columns needed)
# ─────────────────────────────────────────────
def compute_drift(reference, current):
    # drop label columns so drift is computed on features only
    ref = reference.drop(columns=["target", "prediction"], errors="ignore")
    cur = current.drop(columns=["target", "prediction"], errors="ignore")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)   # <-- capture return
    result = snapshot.dict()

    # "share of drifted columns" is the standard drift score in new API
    return extract_metric(result, "drift", "share")


# ─────────────────────────────────────────────
# ACCURACY  (needs a DataDefinition)
# ─────────────────────────────────────────────
def compute_accuracy(reference, current):
    definition = DataDefinition(
        classification=[
            BinaryClassification(target="target", prediction_labels="prediction")
        ]
    )

    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)  # <-- capture
    result = snapshot.dict()

    return extract_metric(result, "accuracy")


# ──────────────────────────────���──────────────
# MAIN
# ─────────────────────────────────────────────
def monitor_model():
    print("=" * 55)
    print(" CUSTOMER CHURN MONITORING ")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_data()
    model = load_model()
    reference, current = prepare(X_train, X_test, y_train, y_test, model)

    logger.info("Computing drift...")
    drift = compute_drift(reference, current)

    logger.info("Computing accuracy...")
    acc = compute_accuracy(reference, current)

    print(f"DRIFT SCORE  : {drift}")
    print(f"ACCURACY     : {acc}")

    DRIFT_GAUGE.set(drift)
    ACC_GAUGE.set(acc)

    print("=" * 55)
    print(" COMPLETED ")
    print("=" * 55)


if __name__ == "__main__":
    start_prometheus()
    monitor_model()