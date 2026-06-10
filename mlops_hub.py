import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ecommerce Customer Churn Control Hub", layout="wide", page_icon="⚙️")

# ----------------------------
# CONFIG
# ----------------------------
AIRFLOW_URL      = "http://localhost:8080"
MLFLOW_URL       = "http://localhost:5000"
USER_APP_URL     = "http://localhost:8501"
DRIFT_REPORT_URL = "http://127.0.0.1:5500/reports/monitoring/xgbclassifier_drift_report.html"
CLASS_REPORT_URL = "http://127.0.0.1:5500/reports/monitoring/xgbclassifier_classification_report.html"
PROMETHEUS_URL   = "http://localhost:8000/metrics"
AIRFLOW_AUTH     = ("airflow", "airflow")

DAG_TRIGGER_API    = f"{AIRFLOW_URL}/api/v1/dags/{{dag_id}}/dagRuns"
DAG_TRIGGER_API_V2 = f"{AIRFLOW_URL}/api/v2/dags/{{dag_id}}/dagRuns"

# ----------------------------
# SERVICES
# ----------------------------
SERVICES = [
    {"id": "airflow_scheduler", "name": "Airflow Scheduler",     "url": None},
    {"id": "airflow_webserver", "name": "Airflow Webserver",     "url": AIRFLOW_URL},
    {"id": "user_app",          "name": "User App",              "url": USER_APP_URL},
    {"id": "data_drift",        "name": "Data Drift",            "url": DRIFT_REPORT_URL},
    {"id": "class_report",      "name": "Classification Report", "url": CLASS_REPORT_URL},
    {"id": "mlflow",            "name": "MLflow",                "url": MLFLOW_URL},
    {"id": "prometheus",        "name": "Prometheus Monitoring", "url": PROMETHEUS_URL},
]

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
<div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
  <h1 style='margin:0; font-size:2rem;'>⚙️ Ecommerce Customer Churn Control Hub</h1>
  <p style='color:gray; margin-top:4px;'>Manage pipelines, services, and monitoring in one place</p>
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
    is_running = st.session_state[key]

    with col:
        status_color = "🟢" if is_running else "🔴"
        status_text  = "Running" if is_running else "Stopped"

        with st.container(border=True):
            st.markdown(f"**{svc['name']}**")
            st.markdown(f"{status_color} {status_text}")

            has_url = svc["url"] is not None

            if has_url:
                btn_cols = st.columns([1, 1])
                with btn_cols[0]:
                    if not is_running:
                        if st.button("▶ Start", key=f"start_{svc['id']}", use_container_width=True):
                            st.session_state[key] = True
                            st.rerun()
                    else:
                        if st.button("⏹ Stop", key=f"stop_{svc['id']}", use_container_width=True):
                            st.session_state[key] = False
                            st.rerun()
                with btn_cols[1]:
                    st.link_button("🔗 Open", svc["url"], use_container_width=True)
            else:
                if not is_running:
                    if st.button("▶ Start", key=f"start_{svc['id']}", use_container_width=True):
                        st.session_state[key] = True
                        st.rerun()
                else:
                    if st.button("⏹ Stop", key=f"stop_{svc['id']}", use_container_width=True):
                        st.session_state[key] = False
                        st.rerun()

st.divider()

# ----------------------------
# METRICS ROW
# ----------------------------
running_count = sum(1 for svc in SERVICES if st.session_state[f"running_{svc['id']}"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Services Running",    f"{running_count} / {len(SERVICES)}")
m2.metric("Data Drift (latest)", "0.45", delta="↑ +0.07", delta_color="inverse")
m3.metric("Model Accuracy",      "82%",  delta="↓ -1%",   delta_color="inverse")
m4.metric("DAGs Triggered",      st.session_state.dag_count)

st.divider()

# ----------------------------
# QUICK LINKS
# ----------------------------
st.subheader("🔗 Quick Access")

q1, q2, q3, q4, q5 = st.columns(5)
q1.link_button("🌀 Airflow UI",            AIRFLOW_URL,      use_container_width=True)
q2.link_button("📊 MLflow UI",             MLFLOW_URL,       use_container_width=True)
q3.link_button("📉 Data Drift Report",     DRIFT_REPORT_URL, use_container_width=True)
q4.link_button("📋 Classification Report", CLASS_REPORT_URL, use_container_width=True)
q5.link_button("🔥 Prometheus Metrics",    PROMETHEUS_URL,   use_container_width=True)

st.divider()

# ----------------------------
# DAG TRIGGER
# ----------------------------
st.subheader("🚀 Trigger Airflow DAG")

with st.container(border=True):
    st.markdown("### ML Pipeline")
    st.write("Trigger the main MLOps pipeline workflow.")

    if st.button("▶ Trigger ML Pipeline", use_container_width=True):
        try:
            resp = requests.post(
                DAG_TRIGGER_API.format(dag_id="ml_pipeline"),
                auth=AIRFLOW_AUTH,
                headers={"Content-Type": "application/json"},
                json={"conf": {}},
                timeout=5,
            )
            # Auto-retry with v2 if v1 returns 405
            if resp.status_code == 405:
                resp = requests.post(
                    DAG_TRIGGER_API_V2.format(dag_id="ml_pipeline"),
                    auth=AIRFLOW_AUTH,
                    headers={"Content-Type": "application/json"},
                    json={"conf": {}},
                    timeout=5,
                )

            st.write("Status Code:", resp.status_code)
            st.write("Response:", resp.text)

            if resp.status_code in [200, 201]:
                st.session_state.dag_count += 1
                st.success("✅ ML Pipeline DAG triggered successfully!")
            elif resp.status_code == 401:
                st.error("❌ Unauthorized — check username/password in AIRFLOW_AUTH")
            elif resp.status_code == 404:
                st.error("❌ DAG 'ml_pipeline' not found — make sure it exists in Airflow")
            elif resp.status_code == 409:
                st.warning("⚠️ DAG is already running or paused")
            else:
                st.error(f"❌ Failed to trigger DAG ({resp.status_code})")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to Airflow. Make sure Airflow is running on localhost:8080.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.divider()

# ----------------------------
# MONITORING CHART
# ----------------------------
st.subheader("📈 Monitoring Dashboard")

df = pd.DataFrame({
    "Step": range(1, 11),
    "Data Drift":     [0.12, 0.15, 0.18, 0.22, 0.30, 0.28, 0.35, 0.40, 0.38, 0.45],
    "Model Accuracy": [0.91, 0.90, 0.89, 0.88, 0.86, 0.87, 0.85, 0.84, 0.83, 0.82],
})

fig = px.line(
    df, x="Step", y=["Data Drift", "Model Accuracy"],
    title="Data Drift vs Model Accuracy over time",
    color_discrete_map={"Data Drift": "#378ADD", "Model Accuracy": "#1D9E75"},
    markers=True,
)
fig.update_layout(legend_title_text="", margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

st.caption("Replace chart data with real Prometheus / MLflow metrics.")
#  streamlit run mlops_hub.py --server.port 8502   