# Credit Card Fraud Detection ML API

An end-to-end Machine Learning project for detecting fraudulent credit card transactions. The project is based on my Master's thesis on designing and evaluating machine learning pipelines for financial applications.

## Project Overview

The goal of this project is to build a fraud detection system while handling challenges such as:

- Highly imbalanced transaction data
- Feature engineering
- Temporal data
- Model evaluation using appropriate metrics
- Model deployment through an API
- Containerization using Docker

## Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized credit card transactions.

The original dataset contains:

- 284,807 transactions
- 31 columns
- 492 fraud transactions
- Severe class imbalance

Duplicate transactions are removed during preprocessing.

The dataset itself is not included in this repository.

## Machine Learning Experiments

I evaluated 60 different ML pipeline combinations:

3 Feature Configurations × 5 Imbalance Strategies × 4 ML Models = 60 Pipelines

### Feature Configurations

- Configuration A: Original transaction features
- Configuration B: Original features + time-based engineered features
- Configuration C: Original features + amount-based engineered features

### Imbalance Strategies

- No imbalance handling
- Class weighting
- Random Undersampling
- Random Oversampling
- SMOTE

### Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

PR-AUC was used as the primary evaluation metric because fraud detection is a highly imbalanced classification problem.

## Selected Pipeline

The thesis evaluation selected:

- Feature Configuration B
- Random Oversampling
- Random Forest

The production version uses the time-enhanced feature configuration and a Random Forest classifier.

## Production Pipeline

The application follows this workflow:

Transaction Data  
→ Feature Engineering  
→ StandardScaler  
→ Random Forest Model  
→ Fraud Prediction

The trained model and preprocessing artifacts are saved using Joblib and loaded by the API for inference.

## FastAPI

The model is exposed through a FastAPI REST API.

Main endpoints:

- `GET /health` - checks whether the API is running
- `POST /predict` - predicts whether a transaction is fraudulent

Example response:

```json
{
  "prediction": 1,
  "fraud_probability": 0.9983
}
```

Where:

- `0` = Normal transaction
- `1` = Fraudulent transaction

## Docker

The application is containerized using Docker.

Build the image:

```bash
docker build -t fraud-detection-api .
```

Run the container:

```bash
docker run -p 8000:8000 fraud-detection-api
```

Open the API documentation:

`http://127.0.0.1:8000/docs`

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib
- Docker

## Project Structure

```text
fraud-detection-ml/
├── data/
├── train.py
├── predict.py
├── main.py
├── model_60pip_config.py
├── fraud_model.pkl
├── scaler.pkl
├── time_max.pkl
├── time_bins.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
