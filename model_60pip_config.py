import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

######Load and clean dataset

df = pd.read_csv(r"C:\Users\pooja\OneDrive\Python\python_program_practice\fraud-detection-ml\data/creditcard.csv")

df = df.drop_duplicates()
df = df.sort_values("Time").reset_index(drop=True)

print("Dataset shape:", df.shape)

#### Separate features and target

X = df.drop("Class", axis=1)
y = df["Class"]

####Chronological 80/20 split
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

##### all the feature configurations
# Configuration A - Original features
X_train_A = X_train.copy()
X_test_A = X_test.copy()

####Configuration B - Time-enhanced features
X_train_B = X_train.copy()
X_test_B = X_test.copy()

time_max = X_train_B["Time"].max()

X_train_B["Time_Norm"] = X_train_B["Time"] / time_max
X_test_B["Time_Norm"] = X_test_B["Time"] / time_max

X_train_B["Hour_Bin"] = (
    (X_train_B["Time"] % 86400) // 3600
)

X_test_B["Hour_Bin"] = (
    (X_test_B["Time"] % 86400) // 3600
)

X_train_B["Time_Segment"] = pd.cut(
    X_train_B["Time"],
    bins=6,
    labels=False
).fillna(0).astype(int)

X_test_B["Time_Segment"] = pd.cut(
    X_test_B["Time"],
    bins=6,
    labels=False
).fillna(0).astype(int)

X_train_B["Day2_Flag"] = (
    X_train_B["Time"] > 86400
).astype(int)

X_test_B["Day2_Flag"] = (
    X_test_B["Time"] > 86400
).astype(int)

#### Configuration C - Amount-enhanced features
X_train_C = X_train.copy()
X_test_C = X_test.copy()

X_train_C["Log_Amount"] = (
    X_train_C["Amount"] + 1
).apply(lambda x: __import__("numpy").log(x))

X_test_C["Log_Amount"] = (
    X_test_C["Amount"] + 1
).apply(lambda x: __import__("numpy").log(x))

X_train_C["Amount_Bin"] = pd.qcut(
    X_train_C["Amount"],
    q=5,
    labels=False,
    duplicates="drop"
)

X_test_C["Amount_Bin"] = pd.qcut(
    X_test_C["Amount"],
    q=5,
    labels=False,
    duplicates="drop"
)

X_train_C["Amount_Sq"] = X_train_C["Amount"] ** 2
X_test_C["Amount_Sq"] = X_test_C["Amount"] ** 2

X_train_C["Amt_Time_Interact"] = (
    X_train_C["Amount"] * X_train_C["Time"]
)

X_test_C["Amt_Time_Interact"] = (
    X_test_C["Amount"] * X_test_C["Time"]
)


feature_sets = {
    "A": (X_train_A, X_test_A),
    "B": (X_train_B, X_test_B),
    "C": (X_train_C, X_test_C)
}

#### all the models used

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )
}

####6. Imbalance strategies

imbalance_strategies = [
    "none",
    "weighted",
    "undersample",
    "oversample",
    "smote"
]

#### Runs all 60 experiments

results = []

for feature_name, (X_train_config, X_test_config) in feature_sets.items():

    # Scale each feature configuration
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_config)
    X_test_scaled = scaler.transform(X_test_config)

    for strategy in imbalance_strategies:

        # Default training data
        X_resampled = X_train_scaled
        y_resampled = y_train

        # Apply imbalance strategy
        if strategy == "oversample":

            sampler = RandomOverSampler(random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(
                X_train_scaled,
                y_train
            )

        elif strategy == "smote":

            sampler = SMOTE(random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(
                X_train_scaled,
                y_train
            )

        elif strategy == "undersample":

            sampler = RandomUnderSampler(random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(
                X_train_scaled,
                y_train
            )

        for model_name, base_model in models.items():

            print(
                "Running:",
                feature_name,
                strategy,
                model_name
            )

            # Create model
            model = base_model

            # Handle class weighting
            if strategy == "weighted":

                if model_name == "Logistic Regression":

                    model = LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )

                elif model_name == "Decision Tree":

                    model = DecisionTreeClassifier(
                        max_depth=10,
                        class_weight="balanced",
                        random_state=42
                    )

                elif model_name == "Random Forest":

                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=10,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    )

                elif model_name == "XGBoost":

                    fraud_count = y_train.value_counts()[1]
                    normal_count = y_train.value_counts()[0]

                    scale_weight = normal_count / fraud_count

                    model = XGBClassifier(
                        n_estimators=50,
                        max_depth=6,
                        scale_pos_weight=scale_weight,
                        random_state=42,
                        eval_metric="logloss"
                    )

            # Train model
            model.fit(
                X_resampled,
                y_resampled
            )

            # Predictions
            predictions = model.predict(X_test_scaled)

            probabilities = model.predict_proba(
                X_test_scaled
            )[:, 1]

            # Evaluation
            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            pr_auc = average_precision_score(
                y_test,
                probabilities
            )

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )

            results.append({
                "Feature_Config": feature_name,
                "Imbalance_Strategy": strategy,
                "Model": model_name,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "PR_AUC": pr_auc,
                "ROC_AUC": roc_auc
            })

##### 8. Compare results


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "PR_AUC",
    ascending=False
)

print("\nTop 10 pipelines:")
print(results_df.head(10))

####experiment results

results_df.to_csv(
    "experiment_results.csv",
    index=False
)

print("\nExperiment results saved!")