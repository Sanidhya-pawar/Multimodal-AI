import streamlit as st
import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from collections import Counter

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Multimodal AI Platform",
    page_icon="🤖",
    layout="wide"
)

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
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "datasets/cleaned_sentiment140.csv"
    )

df = load_data()

# ==========================================
# BUILD VOCAB
# ==========================================

counter = Counter()

for text in df["clean_text"]:
    counter.update(text.split())

vocab = {
    word: idx + 2
    for idx, (word, count)
    in enumerate(counter.most_common(VOCAB_SIZE))
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# ==========================================
# ENCODER
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
# BILSTM
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
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    bilstm = BiLSTM()

    bilstm.load_state_dict(
        torch.load(
            "saved_models/bilstm.pth",
            map_location=DEVICE
        )
    )

    bilstm.eval()

    bert_tokenizer = AutoTokenizer.from_pretrained(
        "saved_models/bert"
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_models/bert"
    )

    deberta_tokenizer = AutoTokenizer.from_pretrained(
        "saved_models/deberta"
    )

    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_models/deberta"
    )

    return (
        bilstm,
        bert_tokenizer,
        bert_model,
        deberta_tokenizer,
        deberta_model
    )

(
    bilstm,
    bert_tokenizer,
    bert_model,
    deberta_tokenizer,
    deberta_model
) = load_models()

# ==========================================
# PREDICT FUNCTIONS
# ==========================================

def predict_bilstm(text):

    encoded = encode_text(text)

    tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    with torch.no_grad():

        prob = bilstm(
            tensor
        ).item()

    label = (
        "Positive"
        if prob > 0.5
        else "Negative"
    )

    return label, prob

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

    with torch.no_grad():

        outputs = model(**encoded)

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
# UI
# ==========================================

st.title("🤖 Multimodal AI Benchmarking Platform")

st.markdown(
    "### Text Sentiment Analysis Module"
)

text_input = st.text_area(
    "Enter text",
    height=150
)

model_choice = st.selectbox(
    "Select Model",
    [
        "BiLSTM",
        "BERT",
        "DeBERTa-v3",
        "Compare All"
    ]
)

if st.button("Predict"):

    if not text_input.strip():

        st.warning(
            "Please enter text."
        )

    else:

        if model_choice == "BiLSTM":

            label, conf = predict_bilstm(
                text_input
            )

            st.success(
                f"{label} ({conf:.4f})"
            )

        elif model_choice == "BERT":

            label, conf = predict_transformer(
                text_input,
                bert_tokenizer,
                bert_model
            )

            st.success(
                f"{label} ({conf:.4f})"
            )

        elif model_choice == "DeBERTa-v3":

            label, conf = predict_transformer(
                text_input,
                deberta_tokenizer,
                deberta_model
            )

            st.success(
                f"{label} ({conf:.4f})"
            )

        else:

            bilstm_result = predict_bilstm(
                text_input
            )

            bert_result = predict_transformer(
                text_input,
                bert_tokenizer,
                bert_model
            )

            deberta_result = predict_transformer(
                text_input,
                deberta_tokenizer,
                deberta_model
            )

            results = pd.DataFrame({
                "Model": [
                    "BiLSTM",
                    "BERT",
                    "DeBERTa-v3"
                ],
                "Prediction": [
                    bilstm_result[0],
                    bert_result[0],
                    deberta_result[0]
                ],
                "Confidence": [
                    bilstm_result[1],
                    bert_result[1],
                    deberta_result[1]
                ]
            })

            st.dataframe(
                results,
                use_container_width=True
            )

# ==========================================
# METRICS SECTION
# ==========================================

st.markdown("---")

st.subheader(
    "Benchmark Results"
)

try:

    metrics = pd.read_csv(
        "results/metrics.csv"
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )

    st.bar_chart(
        metrics.set_index(
            "Model"
        )["Accuracy"]
    )

except:

    st.info(
        "Run evaluate.py first."
    )