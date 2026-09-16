import pandas as pd
import joblib
model = joblib.load(r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/fraud_model.pkl")
scaler = joblib.load(r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/scaler.pkl")
time_max = joblib.load(r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/time_max.pkl")
time_bins = joblib.load(r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/time_bins.pkl")

print("Model and preprocessing files loaded successfully!")

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

##### Take one transaction
######transaction = df.drop("Class", axis=1).iloc[[0]].copy()
# Select one actual fraud transaction
fraud_transaction = df[df["Class"] == 1].iloc[[0]]

# Keep the actual answer so we can compare later
actual_class = fraud_transaction["Class"].iloc[0]

# Remove Class because the model should not see the answer
transaction = fraud_transaction.drop("Class", axis=1).copy()
##print(transaction.iloc[0].to_json())
transaction.iloc[0].to_json("transaction.json", indent=4)

print("transaction.json created!")

print("Actual class:", actual_class)
print(transaction)

print(transaction)

#### feature engg 

# Create the 4 time-based features
transaction["Time_Norm"] = transaction["Time"] / time_max

transaction["Hour_Bin"] = (
    (transaction["Time"] % 86400) // 3600
)

transaction["Day2_Flag"] = (
    transaction["Time"] > 86400
).astype(int)

transaction["Time_Segment"] = pd.cut(
    transaction["Time"],
    bins=time_bins,
    labels=False,
    include_lowest=True
).fillna(0).astype(int)

print("Transaction shape after feature engineering:", transaction.shape)

#### Scale the transaction
transaction_scaled = scaler.transform(transaction)

print("Transaction scaled successfully!")
print("Scaled shape:", transaction_scaled.shape)

##### Make prediction
prediction = model.predict(transaction_scaled)

# Get fraud probability
fraud_probability = model.predict_proba(transaction_scaled)[:, 1]

print("Prediction:", prediction[0])
print("Fraud probability:", fraud_probability[0])