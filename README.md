# Global Mobility Application Analyzer

Production-level end-to-end Machine Learning application for visa application classification using the EasyVisa dataset with FastAPI, Streamlit, SHAP/LIME explainability, Docker, AWS ECR/EC2 deployment, and GitHub Actions CI/CD automation.

---

## Project Overview

The Global Mobility Application Analyzer is a production-ready Machine Learning application designed to predict visa application outcomes using the EasyVisa dataset.

The project covers the complete ML lifecycle:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation
- Prediction pipeline
- Explainability
- FastAPI application
- Streamlit dashboard
- Dockerization
- AWS deployment
- GitHub Actions CI/CD automation

The application predicts whether a visa application will be:

| Output | Meaning |
|---|---|
| Certified | Approved |
| Denied | Rejected |

---

## Problem Statement

Visa approval decisions depend on multiple applicant and employer-related factors such as education, job experience, wage, region, and employment details.

This project uses Machine Learning to classify visa applications and provides prediction probabilities with explainability using SHAP and LIME.

---

## Dataset

Dataset used:

```text
EasyVisa Dataset from Kaggle
```

Target column:

```text
case_status
```

Target mapping:

| Original Value | Encoded Value |
|---|---|
| Certified | 1 |
| Denied | 0 |

---

## Features

### Machine Learning

- Modular Machine Learning pipeline
- Train/test data split
- Schema validation
- Missing value checks
- Duplicate value checks
- Target column validation
- Feature transformation
- Model training and evaluation
- Prediction probability output

### Explainability

- SHAP values
- SHAP summary plot
- SHAP force plot
- LIME explanation
- Feature importance report

### Application

- FastAPI backend application
- Streamlit dashboard
- Swagger UI documentation
- JSON prediction response
- Dockerized FastAPI service
- Dockerized Streamlit service
- AWS cloud deployment
- GitHub Actions CI/CD pipeline

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10.11 |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Explainability | SHAP, LIME |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Containerization | Docker |
| Cloud Services | AWS S3, AWS ECR, AWS EC2 |
| CI/CD | GitHub Actions |
| Operating System | Windows (Development), Ubuntu (EC2) |

---

## Project Structure

```text
Global-Mobility-Application-Analyzer/
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── explainability/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── utils/
│   ├── logger/
│   └── exception/
│
├── .github/
│   └── workflows/
│       └── cicd.yml
│
├── app.py
├── dashboard.py
├── Dockerfile.api
├── Dockerfile.dashboard
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline Architecture

### 1. Data Ingestion

Implemented:

- Raw dataset loading
- Train/test split

Artifacts generated:

```text
raw.csv
train.csv
test.csv
```

---

### 2. Data Validation

Implemented:

- Schema validation
- Missing value checks
- Duplicate checks
- Target validation

Outputs:

```text
Validation reports
Validation logs
```

---

### 3. Data Transformation

Implemented:

- Dropped `case_id`
- Target encoding
- OneHotEncoder
- StandardScaler
- ColumnTransformer pipeline
- Dense matrix conversion

Artifacts generated:

```text
transformed train/test CSV
preprocessor.pkl
```

---

### 4. Model Training and Evaluation

Models trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

Best model:

```text
GradientBoostingClassifier
```

Performance:

| Metric | Score |
|---|---|
| Train F1 Score | 0.8263 |
| Test F1 Score | 0.8289 |

Saved model structure:

```python
{
    "preprocessor": preprocessor,
    "model": trained_model
}
```

Artifacts generated:

```text
model.pkl
model_report.yaml
```

---

### 5. Prediction Pipeline

Implemented:

- `PredictionPipelineConfig`
- `CustomData` class
- `PredictionPipeline` class

Features:

- Load trained model
- Load preprocessor
- Predict class
- Predict probabilities

Sample prediction output:

```json
{
  "prediction": 1,
  "result": "Certified",
  "certified_probability": 0.8116,
  "denied_probability": 0.1884
}
```

---

## FastAPI Application

Main API file:

```text
app.py
```

Implemented endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict visa application outcome |
| GET | `/explainability/latest` | Latest explainability outputs |

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Streamlit Dashboard

Dashboard file:

```text
dashboard.py
```

Implemented:

- Prediction form
- Prediction probabilities
- SHAP visualizations
- LIME visualizations
- Feature importance chart/table
- Auto refresh dashboard

Local Dashboard URL:

```text
http://localhost:8501
```

---

## Explainability

### SHAP Explainability

Implemented:

- SHAP values
- SHAP summary plot
- SHAP force plot
- Feature importance

Artifacts generated:

```text
shap_values.pkl
shap_summary.png
shap_force.html
feature_importance.csv
```

---

### LIME Explainability

Implemented:

- LimeTabularExplainer
- HTML explanation export

Artifact generated:

```text
lime_explanation.html
```

---

## Docker Setup

Two separate Docker containers are used.

| Service | Dockerfile | Docker Image | Container Name |
|---|---|---|---|
| FastAPI API | Dockerfile.api | gmaa-api | gmaa-api-container |
| Streamlit Dashboard | Dockerfile.dashboard | gmaa-dashboard | gmaa-dashboard-container |

Local URLs:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## AWS Deployment

AWS services used:

- AWS S3
- AWS ECR
- AWS EC2

### AWS S3

S3 Bucket:

```text
gmaa-artifacts-637161850571
```

Uploaded artifacts:

```text
artifacts/data_ingestion
artifacts/data_validation
artifacts/data_transformation
artifacts/model_trainer
artifacts/explainability
```

---

### AWS ECR

Repositories created:

```text
gmaa-api
gmaa-dashboard
```

Repository URIs:

```text
637161850571.dkr.ecr.us-east-1.amazonaws.com/gmaa-api
637161850571.dkr.ecr.us-east-1.amazonaws.com/gmaa-dashboard
```

---

### AWS EC2

EC2 Details:

| Item | Value |
|---|---|
| OS | Ubuntu Server |
| Instance Type | t3.small |
| Storage | 30 GB |

Security Group Ports:

| Port | Purpose |
|---|---|
| 22 | SSH |
| 80 | HTTP |
| 8000 | FastAPI |
| 8501 | Streamlit |

---

## Production URLs

### FastAPI Swagger UI

```text
http://13.51.161.182:8000/docs
```

### Streamlit Dashboard

```text
http://13.51.161.182:8501
```

---

## GitHub Actions CI/CD Pipeline

This project includes a complete Docker-based CI/CD pipeline using GitHub Actions, AWS ECR, and AWS EC2.

Workflow file:

```text
.github/workflows/cicd.yml
```

The pipeline is automatically triggered whenever code is pushed to the `main` branch.

### CI/CD Workflow Steps

1. Checkout latest source code from GitHub
2. Configure AWS credentials using GitHub Secrets
3. Login to Amazon ECR
4. Build FastAPI Docker image
5. Build Streamlit Docker image
6. Tag Docker images
7. Push Docker images to ECR
8. SSH into EC2
9. Pull latest Docker images from ECR
10. Stop old running containers
11. Remove old containers
12. Run updated Docker containers
13. Verify deployment using Docker container status

### GitHub Secrets Used

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ACCOUNT_ID
EC2_HOST
EC2_USER
EC2_SSH_KEY
```

Sensitive credentials are securely stored in GitHub Actions Secrets and are never committed to the repository.

### Docker Images Used in CI/CD

| Service | Dockerfile | ECR Repository |
|---|---|---|
| FastAPI API | Dockerfile.api | gmaa-api |
| Streamlit Dashboard | Dockerfile.dashboard | gmaa-dashboard |

### AWS Deployment Flow

```text
GitHub Push
     ↓
GitHub Actions Pipeline
     ↓
Build Docker Images
     ↓
Push Images to AWS ECR
     ↓
SSH into AWS EC2
     ↓
Pull Latest Images
     ↓
Restart Docker Containers
     ↓
Production Deployment Updated
```

---

## Local Setup Instructions

### Clone Repository

```bash
git clone https://github.com/Pratham0119/Global-Mobility-Application-Analyzer.git
```

```bash
cd Global-Mobility-Application-Analyzer
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

---

### Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run FastAPI Locally

```bash
uvicorn app:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

## Run Streamlit Dashboard Locally

```bash
streamlit run dashboard.py
```

Open:

```text
http://localhost:8501
```

---

## Docker Commands

### Build FastAPI Image

```bash
docker build -f Dockerfile.api -t gmaa-api .
```

### Build Streamlit Image

```bash
docker build -f Dockerfile.dashboard -t gmaa-dashboard .
```

### Run FastAPI Container

```bash
docker run -d --name gmaa-api-container -p 8000:8000 gmaa-api
```

### Run Streamlit Container

```bash
docker run -d --name gmaa-dashboard-container -p 8501:8501 gmaa-dashboard
```

---

## Screenshots

### FastAPI Swagger UI

Add FastAPI Swagger UI screenshot here.

### Streamlit Dashboard

Add Streamlit dashboard screenshot here.

### SHAP Explainability

Add SHAP visualization screenshot here.

### LIME Explainability

Add LIME explanation screenshot here.

---

## Future Improvements

- Kubernetes deployment
- Terraform infrastructure automation
- MLflow experiment tracking
- Automated model retraining
- Monitoring and alerting
- Authentication and authorization
- HTTPS with domain name
- Blue-green deployment

---

## Author

Pratham Chavan

---

## License

This project is for educational and portfolio purposes.


## Live Demo

The application was deployed on AWS EC2 using Docker, AWS ECR, and GitHub Actions CI/CD.

Live demo may be unavailable when the EC2 instance is stopped to reduce cloud costs.
