# Start Astro
astro dev restart

# Start MLflow
docker compose -f docker-compose.yml up -d

# Connect MLflow to Astro network
$network = docker inspect ecommerce-customer-churn--prediction-mlops-pipeline_43bb99-scheduler-1 --format '{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}'
docker network connect ecommerce-customer-churn--prediction-mlops-pipeline_43bb99_airflow mlflow

Write-Host "? Airflow  ? http://localhost:8080"
Write-Host "? MLflow   ? http://localhost:5000"
