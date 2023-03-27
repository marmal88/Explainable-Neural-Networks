# MLflow on Gcloud

## 1. Set up glcoud infrastructure
***WARNING*** You should only follow the following steps if you intend to replicate an MLflow server on your own GCS instance.  


### 1.1 Setup gcloud CLI
If you are new to gcloud, follow these steps to setup your gcloud CLI [here](https://cloud.google.com/sdk/docs/install)

To switch between current context and new instance please use:
```bash
# setup menu should be initiated to select appropriate gcloud account and project
gcloud init
```

### 1.2 Create GCP service account
The following sets up a google cloud service account to interact with mlflow server, storage bucket, cloud SQL and project

```bash
# creates a new service-account called "mlflow-tracking-sa" with description "Service Account to run the MLFLow tracking server" and name "MLFlow tracking SA"
gcloud iam service-accounts create mlflow-tracking-sa --description="Service Account to run the MLFLow tracking server" --display-name="MLFlow tracking SA"
```

### 1.3 Create your storage bucket 
MLflow requires a storage location to save artifacts. 

Create and link your storage bucket using the command:
```bash
# name the bucket using export 
export GCP_BUCKET_ID=mlflow-bucket-pneumonia
# creates a new bucket called "mlflow-bucket-pneumonia"
# take note that you might need to have mysql-client installed local computer otherwise use gcloud console cli
gsutil mb gs://$GCP_BUCKET_ID
```


### 1.4 Create sql database using the command:
MLflow also requires an SQL instance for logging parameters, metrics and others. Do note that for cloud SQL storage, you can increase storage size, but you cannot decrease it. Thus it is a best practice to size your sql storage using the cloud monitor.

```bash
# obtains the list of tier "" in your region
gcloud sql tiers list --filter="tier:db-f1-micro AND region:asia-southeast1"
# creates a new sql instance named "mlflow-backend", of tier "db-f1-micro", in region "asia-southeast1", of storage type "SSD"
gcloud sql instances create mlflow-backend --tier=db-f1-micro --region=asia-southeast1 --storage-type=SSD --root-password=123
# creates sql database named "mlflow" using instance name "mlflow-backend"
gcloud sql databases create mlflow --instance=mlflow-backend

```

### 1.5 Authorizing the service account to access both storage bucket and project 

```bash
# obtain your project ID using
gcloud projects list
# export the project ID
export GCP_PROJECT_ID=<PROJECT-ID>
# grant access rights for serviceAccount:service-account-name:role for bucket
gsutil iam ch "serviceAccount:mlflow-tracking-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com:roles/storage.admin" gs://$GCP_BUCKET_ID
# add IAM policy binding to PROJECT-ID
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member="serviceAccount:mlflow-tracking-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com" --role=roles/cloudsql.editor
```

## 2. Setup MLflow


### 2.2 Compute instance
You can view the compute instances [here](https://gcpinstances.doit-intl.com/)



```bash
# Upload the script "start_mlflow_tracking.sh" into gcs bucket using
gcloud storage cp $(pwd)/docs/start_mlflow_tracking.sh gs://$GCP_BUCKET_ID
# create 
gcloud beta compute --project=$GCP_PROJECT_ID instances create mlflow-tracking-server \
 --zone=asia-southeast1-a --machine-type=e2-medium \
 --subnet=default --network-tier=PREMIUM \
 --metadata=startup-script-url=gs://$GCP_PROJECT_ID/scripts/start_mlflow_tracking.sh \
 --maintenance-policy=MIGRATE \
 --service-account=mlflow-tracking-sa@<BUCKET-NAME>.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --tags=mlflow-tracking-server --image=cos-77-12371-1109-0 --image-project=cos-cloud --boot-disk-size=10GB --boot-disk-type=pd-balanced --boot-disk-device-name=mlflow-tracking-server --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any

gcloud compute instances create mlflow-tracking-server \
    --project=$GCP_PROJECT_ID \
    --zone=asia-southeast1-a --machine-type=e2-medium \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --metadata=startup-script-url=gs://$GCP_PROJECT_ID/scripts/start_mlflow_tracking.sh \
    --maintenance-policy=MIGRATE \
    --service-account=mlflow-tracking-sa@$GCP_BUCKET_ID.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=mlflow-tracking-server \
    --image=cos-101-17162-127-42 \
    --image-project=cos-cloud \
    --boot-disk-size=10GB --boot-disk-type=pd-balanced --boot-disk-device-name=mlflow-tracking-server --no-shielded-secure-boot \
    --shielded-vtpm --shielded-integrity-monitoring \
    --reservation-affinity=any
```

### 2.3 Firewall

```bash
gcloud compute firewall-rules create allow-mlflow-tracking \
    --network default \
    --priority 1000 \
    --direction ingress \
    --action allow \
    --target-tags mlflow-tracking-server --source-ranges 0.0.0.0/0 \
    --rules tcp:5000 \
    --enable-logging
```

Training
```bash
export MLFLOW_TRACKING_URI=http://<EXTERNAL-IP>:5000
```