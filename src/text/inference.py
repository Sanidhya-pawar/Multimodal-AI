import torch
import torch.nn as nn
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from collections import Counter

# ==========================================
# CONFIG
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 20000
MAX_LEN = 100

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

# ==========================================
# LOAD DATASET FOR VOCAB
# ==========================================

df = pd.read_csv(
    "datasets/cleaned_sentiment140.csv"
)

texts = df["clean_text"].tolist()

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
# TEXT ENCODING
# ==========================================

def encode_text(text):

    tokens = text.lower().split()

    encoded = [
        vocab.get(token, 1)
        for token in tokens
    ]

    encoded = encoded[:MAX_LEN]

    encoded += [0] * (
        MAX_LEN - len(encoded)
    )

    return encoded

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
# LOAD BILSTM
# ==========================================

bilstm = BiLSTM()

bilstm.load_state_dict(
    torch.load(
        "saved_models/bilstm.pth",
        map_location=DEVICE
    )
)

bilstm.to(DEVICE)

bilstm.eval()

# ==========================================
# LOAD BERT
# ==========================================

bert_tokenizer = AutoTokenizer.from_pretrained(
    "saved_models/bert"
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models/bert"
)

bert_model.to(DEVICE)

bert_model.eval()

# ==========================================
# LOAD DEBERTA
# ==========================================

deberta_tokenizer = AutoTokenizer.from_pretrained(
    "saved_models/deberta"
)

deberta_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models/deberta"
)

deberta_model.to(DEVICE)

deberta_model.eval()

# ==========================================
# BILSTM PREDICT
# ==========================================

def predict_bilstm(text):

    encoded = encode_text(text)

    tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        pred = bilstm(tensor)

        prob = pred.item()

    label = (
        "Positive"
        if prob > 0.5
        else "Negative"
    )

    return label, prob

# ==========================================
# TRANSFORMER PREDICT
# ==========================================

def predict_transformer(
    text,
    tokenizer,
    model
):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        k: v.to(DEVICE)
        for k, v in encoded.items()
    }

    with torch.no_grad():

        outputs = model(**encoded)
        
        print("Raw logits:", outputs.logits)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        )

    label = (
        "Positive"
        if pred.item() == 1
        else "Negative"
    )

    confidence = probs[
        0,
        pred.item()
    ].item()

    return label, confidence

# ==========================================
# MAIN LOOP
# ==========================================

print("=" * 50)
print("TEXT SENTIMENT ANALYZER")
print("=" * 50)

while True:

    text = input(
        "\nEnter text (or 'quit'): "
    )

    if text.lower() == "quit":
        break

    bilstm_result = predict_bilstm(
        text
    )

    bert_result = predict_transformer(
        text,
        bert_tokenizer,
        bert_model
    )

    deberta_result = predict_transformer(
        text,
        deberta_tokenizer,
        deberta_model
    )

    print("\nResults")
    print("-" * 50)

    print(
        f"BiLSTM    : {bilstm_result[0]} "
        f"({bilstm_result[1]:.4f})"
    )

    print(
        f"BERT      : {bert_result[0]} "
        f"({bert_result[1]:.4f})"
    )

    print(
        f"DeBERTa   : {deberta_result[0]} "
        f"({deberta_result[1]:.4f})"
    )