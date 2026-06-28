import joblib
import pandas as pd

# ==========================================
# FEATURE ORDER
# ==========================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# ==========================================
# LOAD MODELS
# ==========================================

random_forest = joblib.load(
    "saved_models/random_forest.pkl"
)

xgboost = joblib.load(
    "saved_models/xgboost.pkl"
)

catboost = joblib.load(
    "saved_models/catboost.pkl"
)

# ==========================================
# INPUT FUNCTION
# ==========================================

def get_user_input():

    values = {}

    print("\nEnter patient details:\n")

    for feature in FEATURES:

        value = float(
            input(f"{feature}: ")
        )

        values[feature] = value

    data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    return data

# ==========================================
# PREDICT FUNCTION
# ==========================================

def predict(model, data):

    pred = model.predict(data)[0]

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(data)[0]

        confidence = max(prob)

    else:

        confidence = None

    label = (
        "Heart Disease"
        if int(pred) == 1
        else "No Heart Disease"
    )

    return label, confidence

# ==========================================
# MAIN
# ==========================================

print("=" * 50)
print("HEART DISEASE PREDICTION")
print("=" * 50)

while True:

    choice = input(
        "\nType 'predict' to continue or 'quit' to exit: "
    )

    if choice.lower() == "quit":
        break

    if choice.lower() != "predict":
        print("Invalid choice.")
        continue

    patient_data = get_user_input()

    rf_result = predict(
        random_forest,
        patient_data
    )

    xgb_result = predict(
        xgboost,
        patient_data
    )

    cat_result = predict(
        catboost,
        patient_data
    )

    print("\nResults")
    print("-" * 50)

    print(
        f"Random Forest : {rf_result[0]} "
        f"({rf_result[1]:.4f})"
    )

    print(
        f"XGBoost       : {xgb_result[0]} "
        f"({xgb_result[1]:.4f})"
    )

    print(
        f"CatBoost      : {cat_result[0]} "
        f"({cat_result[1]:.4f})"
    )