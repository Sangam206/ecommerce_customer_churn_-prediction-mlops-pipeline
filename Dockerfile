FROM astrocrpublic.azurecr.io/runtime:3.2-4
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# FROM apache/airflow:2.10.5 

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt 