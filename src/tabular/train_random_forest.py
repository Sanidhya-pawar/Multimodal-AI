import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("Training Random Forest...\n")

model.fit(
    X_train,
    y_train
)

# ==========================================
# ACCURACY
# ==========================================

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print(f"Accuracy: {acc:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

joblib.dump(
    model,
    "saved_models/random_forest.pkl"
)

print(
    "\nModel saved to saved_models/random_forest.pkl"
)