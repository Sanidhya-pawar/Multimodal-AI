import os
import json
import pickle
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split

# ======================================
# CONFIG
# ======================================

DATA_PATH = "datasets/cleaned_sentiment140.csv"

VOCAB_SIZE = 20000

ARTIFACT_DIR = "artifacts"

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

# ======================================
# LOAD DATA
# ======================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(df.shape)

# ======================================
# TRAIN TEST SPLIT
# ======================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# ======================================
# BUILD VOCAB FROM TRAIN ONLY
# ======================================

counter = Counter()

for text in train_df["clean_text"]:
    counter.update(text.split())

vocab = {
    word: idx + 2
    for idx, (word, count)
    in enumerate(
        counter.most_common(VOCAB_SIZE)
    )
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# ======================================
# SAVE VOCAB
# ======================================

with open(
    f"{ARTIFACT_DIR}/vocab.pkl",
    "wb"
) as f:
    pickle.dump(vocab, f)

# ======================================
# SAVE SPLITS
# ======================================

train_df.to_csv(
    f"{ARTIFACT_DIR}/train.csv",
    index=False
)

test_df.to_csv(
    f"{ARTIFACT_DIR}/test.csv",
    index=False
)

# ======================================
# SAVE LABELS
# ======================================

label_mapping = {
    "negative": 0,
    "positive": 1
}

with open(
    f"{ARTIFACT_DIR}/label_mapping.json",
    "w"
) as f:
    json.dump(
        label_mapping,
        f,
        indent=4
    )

# ======================================
# SAVE CONFIG
# ======================================

config = {
    "vocab_size": VOCAB_SIZE,
    "max_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "random_state": 42
}

with open(
    f"{ARTIFACT_DIR}/config.json",
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=4
    )

print("\nArtifacts created successfully.")