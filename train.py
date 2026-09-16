import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)
from sklearn.metrics import confusion_matrix
import joblib

# Loading  dataset
df = pd.read_csv("data/creditcard.csv")

print("Original shape:", df.shape)

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# Check duplicates values
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Sort transactions by time old transcation for training and new transcartion for testing
df = df.sort_values("Time")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("X shape:", X.shape)
print("y shape:", y.shape)

###chronological split

##### Calculate 80% split point
split_index = int(len(X) * 0.8)

# Split the data
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# Feature engineering

time_max = X_train["Time"].max()

# Normalized time
X_train["Time_Norm"] = X_train["Time"] / time_max
X_test["Time_Norm"] = X_test["Time"] / time_max

# Hour of the day
X_train["Hour_Bin"] = (X_train["Time"] % 86400) // 3600
X_test["Hour_Bin"] = (X_test["Time"] % 86400) // 3600

# First day or second day
X_train["Day2_Flag"] = (X_train["Time"] > 86400).astype(int)
X_test["Day2_Flag"] = (X_test["Time"] > 86400).astype(int)
#### Divide time into 6 segments
### X_train["Time_Segment"] = pd.cut(
###     X_train["Time"], bins=6, labels=False
### )

# ####X_test["Time_Segment"] = pd.cut(
#     X_test["Time"], bins=6, labels=False
# )
# print("Training shape after feature engineering:", X_train.shape)
# print("Testing shape after feature engineering:", X_test.shape)

#######Create 6 time segments from training data
X_train["Time_Segment"], time_bins = pd.cut(
    X_train["Time"],
    bins=6,
    labels=False,
    retbins=True
)

X_train["Time_Segment"] = X_train["Time_Segment"].fillna(0).astype(int)

# Apply the same time segments to test data
X_test["Time_Segment"] = pd.cut(
    X_test["Time"],
    bins=time_bins,
    labels=False,
    include_lowest=True
).fillna(0).astype(int)
# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data scaled:", X_train_scaled.shape)
print("Testing data scaled:", X_test_scaled.shape)

# Random Oversampling
ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(
    X_train_scaled, y_train
)

print("Before oversampling:")
print(y_train.value_counts())

print("After oversampling:")
print(y_train_resampled.value_counts())

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

print("Model training completed!")

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Get fraud probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

##### Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#### Save the trained model
joblib.dump(model, r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml\fraud_model.pkl")

print("Model saved successfully!")
# Save the scaler
joblib.dump(scaler, r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/scaler.pkl")

print("Scaler saved successfully!")
# Save training time maximum
joblib.dump(time_max, r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/time_max.pkl")

print("Time max saved successfully!")

joblib.dump(time_bins, r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml/time_bins.pkl")

print("Time bins saved successfully!")