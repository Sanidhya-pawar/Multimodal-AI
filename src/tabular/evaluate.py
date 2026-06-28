import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "datasets/tabular/heart.csv"
)

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# METRIC FUNCTION
# ==========================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

# ==========================================
# EVALUATE FUNCTION
# ==========================================

def evaluate_model(model_path, model_name):

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    start = time.time()

    predictions = model.predict(X_test)

    end = time.time()

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    metrics["Model"] = model_name

    metrics["Inference_Time"] = end - start

    return metrics

# ==========================================
# EVALUATE ALL MODELS
# ==========================================

results = []

results.append(
    evaluate_model(
        "saved_models/random_forest.pkl",
        "Random Forest"
    )
)

results.append(
    evaluate_model(
        "saved_models/xgboost.pkl",
        "XGBoost"
    )
)

results.append(
    evaluate_model(
        "saved_models/catboost.pkl",
        "CatBoost"
    )
)

# ==========================================
# RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Inference_Time"
    ]
]

print("\nTabular Benchmark Results\n")
print(results_df)

# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/tabular_metrics.csv",
    index=False
)

print(
    "\nSaved to results/tabular_metrics.csv"
)