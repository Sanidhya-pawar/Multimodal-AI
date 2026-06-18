import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import Counter

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "datasets/cleaned_sentiment140.csv"

VOCAB_SIZE = 20000
MAX_LEN = 100

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

BATCH_SIZE = 256
EPOCHS = 5

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading artifacts...")

train_df = pd.read_csv(
    "artifacts/train.csv"
)

test_df = pd.read_csv(
    "artifacts/test.csv"
)

train_texts = train_df["clean_text"].tolist()
train_labels = train_df["sentiment"].tolist()

test_texts = test_df["clean_text"].tolist()
test_labels = test_df["sentiment"].tolist()

# ==========================================
# BUILD VOCAB
# ==========================================

print("Loading vocabulary...")

with open(
    "artifacts/vocab.pkl",
    "rb"
) as f:

    vocab = pickle.load(f)

# ==========================================
# ENCODING
# ==========================================

def encode_text(text):

    tokens = text.split()

    encoded = [
        vocab.get(token, 1)
        for token in tokens
    ]

    if len(encoded) > MAX_LEN:
        encoded = encoded[:MAX_LEN]

    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))

    return encoded

X_train = [
    encode_text(text)
    for text in train_texts
]

X_test = [
    encode_text(text)
    for text in test_texts
]

y_train = train_labels
y_test = test_labels

# ==========================================
# TRAIN TEST SPLIT
# ==========================================



# ==========================================
# DATASET
# ==========================================

class SentimentDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )

train_dataset = SentimentDataset(
    X_train,
    y_train
)

test_dataset = SentimentDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# ==========================================
# MODEL
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

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden = torch.cat(
            (
                hidden_forward,
                hidden_backward
            ),
            dim=1
        )

        output = self.fc(hidden)

        return self.sigmoid(output)

# ==========================================
# MODEL INIT
# ==========================================

model = BiLSTM().to(DEVICE)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("Training started...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(inputs).squeeze()

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Loss: {avg_loss:.4f}"
    )

# ==========================================
# EVALUATION
# ==========================================

print("Evaluating...")

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(DEVICE)

        outputs = model(inputs)

        preds = (
            outputs.squeeze()
            .cpu()
            .numpy()
            > 0.5
        ).astype(int)

        predictions.extend(preds)

        actuals.extend(
            labels.numpy()
        )

accuracy = accuracy_score(
    actuals,
    predictions
)

precision = precision_score(
    actuals,
    predictions
)

recall = recall_score(
    actuals,
    predictions
)

f1 = f1_score(
    actuals,
    predictions
)

print("\nResults")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "saved_models/bilstm.pth"
)

print(
    "\nModel saved to saved_models/bilstm.pth"
)