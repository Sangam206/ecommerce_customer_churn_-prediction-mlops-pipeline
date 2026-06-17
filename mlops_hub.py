import streamlit as st
import requests

st.set_page_config(
    page_title="Ecommerce Customer Churn Control Hub",
    layout="wide",
    page_icon="⚙️"
)

# ----------------------------
# CONFIG
# ----------------------------
AIRFLOW_URL = "http://localhost:8080"
MLFLOW_URL = "http://localhost:5000"
USER_APP_URL = "http://localhost:8501"

DRIFT_REPORT_URL = "http://127.0.0.1:5500/reports/monitoring/xgbclassifier_drift_report.html"
CLASS_REPORT_URL = "http://127.0.0.1:5500/reports/monitoring/xgbclassifier_classification_report.html"
PROMETHEUS_URL = "http://localhost:8000/metrics"

AIRFLOW_AUTH = ("airflow", "airflow")

DAG_TRIGGER_API = f"{AIRFLOW_URL}/api/v1/dags/{{dag_id}}/dagRuns"
DAG_TRIGGER_API_V2 = f"{AIRFLOW_URL}/api/v2/dags/{{dag_id}}/dagRuns"

# ----------------------------
# SERVICES
# ----------------------------
SERVICES = [
    {"id": "airflow_scheduler", "name": "Airflow Scheduler", "url": None},
    {"id": "airflow_webserver", "name": "Airflow Webserver", "url": AIRFLOW_URL},
    {"id": "user_app", "name": "User App", "url": USER_APP_URL},
    {"id": "data_drift", "name": "Data Drift", "url": DRIFT_REPORT_URL},
    {"id": "class_report", "name": "Classification Report", "url": CLASS_REPORT_URL},
    {"id": "mlflow", "name": "MLflow", "url": MLFLOW_URL},
    {"id": "prometheus", "name": "Prometheus Monitoring", "url": PROMETHEUS_URL},
]

# ----------------------------
# SESSION STATE INIT
# ----------------------------
for svc in SERVICES:
    key = f"running_{svc['id']}"
    if key not in st.session_state:
        st.session_state[key] = False

if "dag_count" not in st.session_state:
    st.session_state.dag_count = 0

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div style='text-align:center; padding: 1rem 0;'>
  <h1>⚙️ Ecommerce Customer Churn Control Hub</h1>
  <p style='color:gray;'>Manage pipelines, services, and monitoring in one place</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------
# SERVICES GRID
# ----------------------------
st.subheader("🖥️ Services")

cols = st.columns(3)

for i, svc in enumerate(SERVICES):
    col = cols[i % 3]
    key = f"running_{svc['id']}"

    with col:
        status = "🟢 Running" if st.session_state[key] else "🔴 Stopped"

        with st.container(border=True):
            st.markdown(f"**{svc['name']}**")
            st.markdown(status)

            c1, c2 = st.columns(2)

            if st.button(
                "▶ Start" if not st.session_state[key] else "⏹ Stop",
                key=f"btn_{svc['id']}",
                use_container_width=True
            ):
                st.session_state[key] = not st.session_state[key]

            if svc["url"]:
                st.link_button("🔗 Open", svc["url"], use_container_width=True)

st.divider()

# ----------------------------
# METRICS
# ----------------------------
running_count = sum(st.session_state[f"running_{s['id']}"] for s in SERVICES)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Services Running", f"{running_count}/{len(SERVICES)}")
m2.metric("Data Drift", "0.45", delta="+0.07")
m3.metric("Model Accuracy", "82%", delta="-1%")
m4.metric("DAGs Triggered", st.session_state.dag_count)

st.divider()

# ----------------------------
# QUICK LINKS
# ----------------------------
st.subheader("🔗 Quick Access")

c1, c2, c3, c4, c5 = st.columns(5)
c1.link_button("Airflow", AIRFLOW_URL)
c2.link_button("MLflow", MLFLOW_URL)
c3.link_button("Drift Report", DRIFT_REPORT_URL)
c4.link_button("Classification", CLASS_REPORT_URL)
c5.link_button("Prometheus", PROMETHEUS_URL)

st.divider()

# ----------------------------
# DAG TRIGGER
# ----------------------------
st.subheader("🚀 Trigger Airflow DAG")

with st.container(border=True):
    st.write("Trigger ML pipeline workflow")

    if st.button("▶ Trigger ML Pipeline"):
        try:
            url = DAG_TRIGGER_API.format(dag_id="ml_pipeline")

            resp = requests.post(
                url,
                auth=AIRFLOW_AUTH,
                json={"conf": {}},
                timeout=10
            )

            # fallback to v2
            if resp.status_code == 405:
                resp = requests.post(
                    DAG_TRIGGER_API_V2.format(dag_id="ml_pipeline"),
                    auth=AIRFLOW_AUTH,
                    json={"conf": {}},
                    timeout=10
                )

            st.write("Status:", resp.status_code)
            st.write("Response:", resp.text)

            if resp.status_code in (200, 201):
                st.session_state.dag_count += 1
                st.success("DAG triggered successfully")
            elif resp.status_code == 401:
                st.error("Unauthorized: check Airflow credentials")
            elif resp.status_code == 404:
                st.error("DAG not found")
            elif resp.status_code == 409:
                st.warning("DAG already running")
            else:
                st.error("Failed to trigger DAG")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Airflow at localhost:8080")
        except Exception as e:
            st.error(f"Error: {e}")
#  streamlit run mlops_hub.py --server.port 8502   