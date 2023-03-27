#!/bin/bash
MLFLOW_IMAGE=ghcr.io/mlflow/mlflow
CLOUD_SQL_PROXY_IMAGE=gcr.io/cloudsql-docker/gce-proxy:1.19.1
MYSQL_INSTANCE=$GCP_PROJECT_ID:asia-southeast1:mlflow-backend
GCP_BUCKET_ID=mlflow-bucket-pneumonia

echo 'Starting Cloud SQL Proxy'
docker run -d --name mysql  --net host -p 3306:3306 $CLOUD_SQL_PROXY_IMAGE /cloud_sql_proxy -instances $MYSQL_INSTANCE=tcp:0.0.0.0:3306

echo 'Starting mlflow-tracking server'
docker run -d --net=host -p 5000:5000 \
    --name mlflow-tracking \
    $MLFLOW_IMAGE mlflow server \
    --backend-store-uri mysql+pymysql://root:123@localhost/mlflow \
    --default-artifact-root gs://$GCP_BUCKET_ID/mlflow_artifacts/ --host 0.0.0.0

echo 'Altering IPTables'
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT