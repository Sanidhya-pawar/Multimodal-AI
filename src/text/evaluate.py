import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from collections import Counter

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "datasets/cleaned_sentiment140.csv"

VOCAB_SIZE = 20000
MAX_LEN = 100

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading test artifact...")

test_df = pd.read_csv(
    "artifacts/test.csv"
)

texts = test_df["clean_text"].tolist()
labels = test_df["sentiment"].tolist()

# ==========================================
# BUILD VOCAB
# ==========================================

counter = Counter()

for text in texts:
    counter.update(text.split())

vocab = {
    word: idx + 2
    for idx, (word, count)
    in enumerate(counter.most_common(VOCAB_SIZE))
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# ==========================================
# ENCODING
# ==========================================

def encode_text(text):

    tokens = text.split()

    encoded = [
        vocab.get(token, 1)
        for token in tokens
    ]

    encoded = encoded[:MAX_LEN]

    encoded += [0] * (
        MAX_LEN - len(encoded)
    )

    return encoded

X_test = [
    encode_text(text)
    for text in texts
]

y_test = labels

# ==========================================
# BILSTM MODEL
# ==========================================

class BiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            len(vocab),
            EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            HIDDEN_DIM * 2,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        hidden = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        x = self.fc(hidden)

        return self.sigmoid(x)

# ==========================================
# METRIC FUNCTION
# ==========================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "Accuracy":
        accuracy_score(y_true, y_pred),

        "Precision":
        precision_score(y_true, y_pred),

        "Recall":
        recall_score(y_true, y_pred),

        "F1":
        f1_score(y_true, y_pred)
    }

# ==========================================
# RESULTS LIST
# ==========================================

results = []

# ==========================================
# BILSTM EVALUATION
# ==========================================

print("\nEvaluating BiLSTM...")

bilstm = BiLSTM()

bilstm.load_state_dict(
    torch.load(
        "saved_models/bilstm.pth",
        map_location=DEVICE
    )
)

bilstm.to(DEVICE)

bilstm.eval()

start = time.time()

predictions = []

batch_size = 128

with torch.no_grad():

    for i in range(
        0,
        len(X_test),
        batch_size
    ):

        batch = X_test[
            i:i + batch_size
        ]

        inputs = torch.tensor(
            batch,
            dtype=torch.long
        ).to(DEVICE)

        outputs = bilstm(inputs)

        preds = (
            outputs.squeeze()
            .cpu()
            .numpy()
            > 0.5
        ).astype(int)

        predictions.extend(preds)

end = time.time()

metrics = calculate_metrics(
    y_test,
    predictions
)

metrics["Model"] = "BiLSTM"
metrics["Inference_Time"] = end - start

results.append(metrics)

# ==========================================
# HUGGINGFACE EVALUATION
# ==========================================

def evaluate_transformer(
    model_path,
    model_name
):

    print(
        f"\nEvaluating {model_name}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)

    model.eval()

    sample_texts = test_df[
        "clean_text"
    ].tolist()

    batch_size = 64

    predictions = []

    start = time.time()

    with torch.no_grad():

        for i in range(
            0,
            len(sample_texts),
            batch_size
        ):

            batch = sample_texts[
                i:i + batch_size
            ]

            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            encoded = {
                k: v.to(DEVICE)
                for k, v
                in encoded.items()
            }

            outputs = model(**encoded)

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

    end = time.time()

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    metrics["Model"] = model_name

    metrics["Inference_Time"] = (
        end - start
    )

    results.append(metrics)

# ==========================================
# BERT
# ==========================================

evaluate_transformer(
    "saved_models/bert",
    "BERT"
)

# ==========================================
# DEBERTA
# ==========================================

evaluate_transformer(
    "saved_models/deberta",
    "DeBERTa-v3"
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

print("\nBenchmark Results")

print(results_df)

# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/metrics.csv",
    index=False
)

best_model = results_df.sort_values(
    by="F1",
    ascending=False
).iloc[0]

print("\nBest Model")

print(best_model)

print(
    "\nSaved results to results/metrics.csv"
)