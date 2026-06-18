import pandas as pd
import re
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "datasets/training.1600000.processed.noemoticon.csv"
OUTPUT_PATH = "datasets/cleaned_sentiment140.csv"

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "tweet"
]

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=columns
)

print(f"Dataset Shape: {df.shape}")

# ==========================================
# KEEP REQUIRED COLUMNS
# ==========================================

df = df[["sentiment", "tweet"]]

# ==========================================
# CONVERT TARGET LABELS
# 0 = Negative
# 4 = Positive
# ==========================================

df["sentiment"] = df["sentiment"].map({
    0: 0,
    4: 1
})

# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol only
    text = re.sub(r'#', '', text)

    # Remove RT
    text = re.sub(r'\bRT\b', '', text)

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ==========================================
# CLEAN TWEETS
# ==========================================

print("Cleaning tweets...")

tqdm.pandas()

df["clean_text"] = df["tweet"].progress_apply(clean_text)

# ==========================================
# REMOVE EMPTY TEXT
# ==========================================

df = df[df["clean_text"].str.len() > 0]

# ==========================================
# REMOVE DUPLICATES
# ==========================================

before = len(df)

df.drop_duplicates(
    subset=["clean_text", "sentiment"],
    inplace=True
)

after = len(df)

print(f"Removed {before - after} duplicate rows")

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\nClass Distribution")

print(df["sentiment"].value_counts())

# ==========================================
# SAVE CLEAN DATA
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPreprocessing Complete")
print(f"Saved to: {OUTPUT_PATH}")

print("\nSample Data")

print(df[["sentiment", "clean_text"]].head())