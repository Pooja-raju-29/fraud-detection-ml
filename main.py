### checking if its runing 
# # from fastapi import FastAPI
# app = FastAPI()
# @app.get("/")
# def home():
#     return {"message": "Fraud Detection API is running"}

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI()
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
# Load trained model and preprocessing files
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
time_max = joblib.load("time_max.pkl")
time_bins = joblib.load("time_bins.pkl")
##### to check if its opening 
# @app.post("/predict")
# def predict(transaction: Transaction):
#     data = pd.DataFrame([transaction.model_dump()])

#     return {"message": "Transaction received successfully"}

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert incoming transaction to DataFrame
    data = pd.DataFrame([transaction.model_dump()])

    # Create the same 4 features used during training
    data["Time_Norm"] = data["Time"] / time_max

    data["Hour_Bin"] = (
        (data["Time"] % 86400) // 3600
    )

    data["Day2_Flag"] = (
        data["Time"] > 86400
    ).astype(int)

    data["Time_Segment"] = pd.cut(
        data["Time"],
        bins=time_bins,
        labels=False,
        include_lowest=True
    ).fillna(0).astype(int)

    print("API transaction shape:", data.shape)

    # Scale the transaction
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)

    # Get fraud probability
    fraud_probability = model.predict_proba(data_scaled)[:, 1]

    # Return the result
    return {
        "prediction": int(prediction[0]),
        "fraud_probability": float(fraud_probability[0])
    }

