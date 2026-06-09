import pickle
import hashlib
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from prometheus_client import Gauge, start_http_server
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "after encoding"
SPLIT_DIR    = PROJECT_ROOT / "splitting data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "monitoring"
LOGS_DIR     = PROJECT_ROOT / "logs"
HASH_LOG     = LOGS_DIR / "model_hashes.csv"

X_TRAIN_PATH = DATA_DIR  / "x_train.csv"
X_TEST_PATH  = DATA_DIR  / "x_test.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_TEST_PATH  = SPLIT_DIR / "y_test.csv"
MODEL_PATH   = MODELS_DIR / "model.pkl"

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
DRIFT_THRESHOLD    = 0.5
ACCURACY_THRESHOLD = 0.75

# ─────────────────────────────────────────────
# PROMETHEUS GAUGES
# ─────────────────────────────────────────────
DRIFT_GAUGE         = Gauge("data_drift_share",   "Share of drifted features")
ACC_GAUGE           = Gauge("model_accuracy",     "Model accuracy")
F1_GAUGE            = Gauge("model_f1_macro",     "Model F1 macro")
PRECISION_GAUGE     = Gauge("model_precision",    "Model precision macro")
RECALL_GAUGE        = Gauge("model_recall",       "Model recall macro")
FILE_OK_GAUGE       = Gauge("model_file_ok",      "model.pkl healthy (1=ok, 0=error)")
FILE_CHANGED_GAUGE  = Gauge("model_file_changed", "model.pkl changed since last run (1=yes)")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PKL FILE MONITORING
# ─────────────────────────────────────────────
def get_file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_model_file(path: Path) -> bool:
    """Check existence and corruption of pkl file."""
    # 1. Existence check
    if not path.exists():
        logger.error(f"[ALERT] model.pkl MISSING at {path}")
        return False

    # 2. Size and modified time
    size_kb  = path.stat().st_size / 1024
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    logger.info(f"model.pkl | Size: {size_kb:.1f} KB | Last Modified: {modified}")

    # 3. Corruption check
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        logger.info("model.pkl — Loads successfully. No corruption.")
    except Exception as e:
        logger.error(f"[ALERT] model.pkl CORRUPTED: {e}")
        return False

    return True


def detect_model_change(path: Path) -> bool:
    """Detect if pkl file changed since last run via hash comparison."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current_hash = get_file_hash(path)
    changed      = False

    if HASH_LOG.exists():
        df   = pd.read_csv(HASH_LOG)
        last = df[df["model_name"] == "model"]
        if not last.empty:
            prev_hash = last.iloc[-1]["hash"]
            if current_hash != prev_hash:
                logger.warning(
                    f"[MODEL CHANGED] model.pkl hash changed! "
                    f"Previous: {prev_hash[:8]}... → Current: {current_hash[:8]}... "
                    f"Model was retrained or replaced."
                )
                changed = True
            else:
                logger.info("model.pkl — Hash unchanged. Same model as last run.")

    # Log current hash
    record = pd.DataFrame([{
        "model_name": "model",
        "hash":       current_hash,
        "size_kb":    round(path.stat().st_size / 1024, 2),
        "timestamp":  datetime.now().isoformat()
    }])
    record.to_csv(HASH_LOG, mode="a", header=not HASH_LOG.exists(), index=False)
    logger.info(f"model.pkl — Current hash: {current_hash[:8]}... logged.")

    return changed


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data() -> tuple:
    logger.info("Loading train/test data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test  = pd.read_csv(Y_TEST_PATH).squeeze()
    logger.info(f"Reference (train) rows : {len(X_train)}")
    logger.info(f"Current   (test)  rows : {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    logger.info("Loading model.pkl...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {type(model).__name__}")
    return model


# ─────────────────────────────────────────────
# PREPARE
# ─────────────────────────────────────────────
def prepare(X_train, X_test, y_train, y_test, model) -> tuple:
    reference = X_train.copy()
    reference["target"]     = y_train.values
    reference["prediction"] = model.predict(X_train)

    current = X_test.copy()
    current["target"]     = y_test.values
    current["prediction"] = model.predict(X_test)

    return reference, current


# ─────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────
def compute_metrics(current: pd.DataFrame) -> dict:
    y_true = current["target"].values
    y_pred = current["prediction"].values
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


# ─────────────────────────────────────────────
# ALERTING
# ─────────────────────────────────────────────
def check_alerts(model_name: str, metrics: dict, drift: float = None):
    if drift is not None and drift > DRIFT_THRESHOLD:
        logger.warning(
            f"[ALERT] {model_name} — High data drift! "
            f"Score {drift:.2f} exceeds threshold {DRIFT_THRESHOLD}."
        )
    if metrics["accuracy"] < ACCURACY_THRESHOLD:
        logger.warning(
            f"[ALERT] {model_name} — Low accuracy! "
            f"{metrics['accuracy']:.2f} below threshold {ACCURACY_THRESHOLD}. "
            f"Consider retraining."
        )


# ─────────────────────────────────────────────
# EVIDENTLY HTML REPORT
# ─────────────────────────────────────────────
def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> float:
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently not installed. Run: pip install evidently==0.4.33")
        return 0.0

    try:
        # Features only for drift
        ref_features = reference.drop(columns=["target", "prediction"], errors="ignore")
        cur_features = current.drop(columns=["target", "prediction"],   errors="ignore")

        # Drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=ref_features, current_data=cur_features)

        # Classification report
        class_report = Report(metrics=[ClassificationPreset()])
        class_report.run(reference_data=reference, current_data=current)

        # Save HTML reports
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        drift_path = REPORTS_DIR / f"{name}_drift_report.html"
        class_path = REPORTS_DIR / f"{name}_classification_report.html"

        drift_report.save_html(str(drift_path))
        class_report.save_html(str(class_path))

        logger.info(f"{name} — Drift report          → {drift_path}")
        logger.info(f"{name} — Classification report → {class_path}")

        # Extract drift score
        drift_score = 0.0
        try:
            drift_json  = drift_report.as_dict()
            drift_score = drift_json["metrics"][0]["result"]["drift_share"]
        except Exception:
            logger.warning(f"{name} — Could not extract drift score from report.")

        return drift_score

    except Exception as e:
        logger.error(f"Failed to generate Evidently report for {name}: {e}")
        return 0.0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_monitoring():
    logger.info("=" * 55)
    logger.info(" CUSTOMER CHURN MONITORING")
    logger.info("=" * 55)

    # ── PKL File Monitoring ───────────────────
    logger.info("-" * 55)
    logger.info(" PKL FILE CHECKS")
    logger.info("-" * 55)

    file_ok = check_model_file(MODEL_PATH)
    FILE_OK_GAUGE.set(1 if file_ok else 0)

    if not file_ok:
        logger.error("model.pkl is missing or corrupted. Aborting monitoring.")
        return

    changed = detect_model_change(MODEL_PATH)
    FILE_CHANGED_GAUGE.set(1 if changed else 0)

    # ── Load Data & Model ─────────────────────
    X_train, X_test, y_train, y_test = load_data()
    model     = load_model()
    model_name = type(model).__name__

    # ── Monitoring ────────────────────────────
    logger.info("-" * 55)
    logger.info(f" {model_name} Monitoring")
    logger.info("-" * 55)

    reference, current = prepare(X_train, X_test, y_train, y_test, model)
    metrics            = compute_metrics(current)
    drift              = run_evidently_report(reference, current, model_name.lower())

    logger.info(f"Drift Score : {drift}")
    logger.info(f"Accuracy    : {metrics['accuracy']}")
    logger.info(f"F1 Macro    : {metrics['f1_macro']}")
    logger.info(f"Precision   : {metrics['precision']}")
    logger.info(f"Recall      : {metrics['recall']}")

    check_alerts(model_name, metrics, drift)

    DRIFT_GAUGE.set(drift)
    ACC_GAUGE.set(metrics["accuracy"])
    F1_GAUGE.set(metrics["f1_macro"])
    PRECISION_GAUGE.set(metrics["precision"])
    RECALL_GAUGE.set(metrics["recall"])

    logger.info("=" * 55)
    logger.info(" Monitoring Completed")
    logger.info("=" * 55)


if __name__ == "__main__":
    start_http_server(8000)
    logger.info("Prometheus running → http://localhost:8000/metrics")
    run_monitoring()