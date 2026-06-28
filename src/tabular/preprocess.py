import pandas as pd

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "datasets/tabular/heart.csv"

# ==========================================
# LOAD DATA
# ==========================================

print("Loading Heart Disease dataset...\n")

df = pd.read_csv(DATA_PATH)

# ==========================================
# BASIC INFO
# ==========================================

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# ==========================================
# TARGET CHECK
# ==========================================

if "target" in df.columns:
    target_col = "target"
elif "output" in df.columns:
    target_col = "output"
else:
    target_col = df.columns[-1]

print("\nDetected Target Column:")
print(target_col)

print("\nTarget Distribution:")
print(df[target_col].value_counts())

print("\nPreprocessing check completed successfully.")