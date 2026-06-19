import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import transforms, models

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Multimodal AI Platform",
    page_icon="🤖",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# TEXT CONFIG
# ==========================================

VOCAB_SIZE = 20000
MAX_LEN = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 128

# ==========================================
# IMAGE CONFIG
# ==========================================

IMAGE_CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# ==========================================
# LOAD TEXT DATA + VOCAB
# ==========================================

@st.cache_data
def load_text_data():
    return pd.read_csv("datasets/cleaned_sentiment140.csv")

df = load_text_data()

counter = Counter()

for text in df["clean_text"]:
    counter.update(str(text).split())

vocab = {
    word: idx + 2
    for idx, (word, count)
    in enumerate(counter.most_common(VOCAB_SIZE))
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_text(text):
    tokens = text.lower().split()

    encoded = [
        vocab.get(token, 1)
        for token in tokens
    ]

    encoded = encoded[:MAX_LEN]

    encoded += [0] * (MAX_LEN - len(encoded))

    return encoded

# ==========================================
# TEXT MODEL: BILSTM
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

        self.fc = nn.Linear(HIDDEN_DIM * 2, 1)

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
# IMAGE MODEL: CNN
# ==========================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            64 * 37 * 37,
            128
        )

        self.fc2 = nn.Linear(
            128,
            6
        )

    def forward(self, x):
        x = self.pool(
            self.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            self.relu(
                self.conv2(x)
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

# ==========================================
# LOAD ALL MODELS
# ==========================================

@st.cache_resource
def load_models():
    # Text - BiLSTM
    bilstm = BiLSTM()

    bilstm.load_state_dict(
        torch.load(
            "saved_models/bilstm.pth",
            map_location=DEVICE
        )
    )

    bilstm.to(DEVICE)
    bilstm.eval()

    # Text - BERT
    bert_tokenizer = AutoTokenizer.from_pretrained(
        "saved_models/bert"
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_models/bert"
    )

    bert_model.to(DEVICE)
    bert_model.eval()

    # Image - CNN
    cnn_model = CNN()

    cnn_model.load_state_dict(
        torch.load(
            "saved_models/image_cnn.pth",
            map_location=DEVICE
        )
    )

    cnn_model.to(DEVICE)
    cnn_model.eval()

    # Image - ResNet50
    resnet_model = models.resnet50()

    resnet_model.fc = nn.Linear(
        resnet_model.fc.in_features,
        6
    )

    resnet_model.load_state_dict(
        torch.load(
            "saved_models/resnet50.pth",
            map_location=DEVICE
        )
    )

    resnet_model.to(DEVICE)
    resnet_model.eval()

    return (
        bilstm,
        bert_tokenizer,
        bert_model,
        cnn_model,
        resnet_model
    )

(
    bilstm,
    bert_tokenizer,
    bert_model,
    cnn_model,
    resnet_model
) = load_models()

# ==========================================
# TEXT PREDICTION FUNCTIONS
# ==========================================

def predict_bilstm(text):
    encoded = encode_text(text)

    tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        prob = bilstm(tensor).item()

    label = "Positive" if prob > 0.5 else "Negative"

    return label, prob

def predict_bert(text):
    encoded = bert_tokenizer(
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
        outputs = bert_model(**encoded)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        )

    label = "Positive" if pred.item() == 1 else "Negative"

    confidence = probs[
        0,
        pred.item()
    ].item()

    return label, confidence

# ==========================================
# IMAGE PREDICTION FUNCTIONS
# ==========================================

cnn_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image, model, transform):
    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    label = IMAGE_CLASSES[
        pred.item()
    ]

    return label, confidence.item()

# ==========================================
# UI
# ==========================================

st.title("🤖 Multimodal AI Benchmarking Platform")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Module",
    [
        "Home",
        "Text Sentiment Analysis",
        "Image Classification",
        "Model Comparison"
    ]
)

# ==========================================
# HOME
# ==========================================

if page == "Home":
    st.header("Multimodal AI Benchmarking Platform")

    st.write(
        """
        This project compares machine learning and deep learning models
        across multiple data modalities. The current implementation includes
        Text Sentiment Analysis and Image Classification.
        """
    )

    st.subheader("Implemented Modules")

    st.write("✅ Text Module: BiLSTM and BERT")
    st.write("✅ Image Module: CNN and ResNet50")
    st.write("✅ Evaluation Metrics and Streamlit Dashboard")

    st.info(f"Running on device: {DEVICE}")

# ==========================================
# TEXT PAGE
# ==========================================

elif page == "Text Sentiment Analysis":
    st.header("📝 Text Sentiment Analysis")

    text_input = st.text_area(
        "Enter text",
        height=150
    )

    model_choice = st.selectbox(
        "Select Text Model",
        [
            "BiLSTM",
            "BERT",
            "Compare Both"
        ]
    )

    if st.button("Predict Text Sentiment"):
        if not text_input.strip():
            st.warning("Please enter text.")
        else:
            if model_choice == "BiLSTM":
                label, conf = predict_bilstm(text_input)

                st.success(
                    f"BiLSTM Prediction: {label} ({conf:.4f})"
                )

            elif model_choice == "BERT":
                label, conf = predict_bert(text_input)

                st.success(
                    f"BERT Prediction: {label} ({conf:.4f})"
                )

            else:
                bilstm_result = predict_bilstm(text_input)
                bert_result = predict_bert(text_input)

                results = pd.DataFrame({
                    "Model": [
                        "BiLSTM",
                        "BERT"
                    ],
                    "Prediction": [
                        bilstm_result[0],
                        bert_result[0]
                    ],
                    "Confidence": [
                        bilstm_result[1],
                        bert_result[1]
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True
                )

# ==========================================
# IMAGE PAGE
# ==========================================

elif page == "Image Classification":
    st.header("🖼️ Image Classification")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    model_choice = st.selectbox(
        "Select Image Model",
        [
            "CNN",
            "ResNet50",
            "Compare Both"
        ]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Predict Image Class"):
            if model_choice == "CNN":
                label, conf = predict_image(
                    image,
                    cnn_model,
                    cnn_transform
                )

                st.success(
                    f"CNN Prediction: {label} ({conf:.4f})"
                )

            elif model_choice == "ResNet50":
                label, conf = predict_image(
                    image,
                    resnet_model,
                    resnet_transform
                )

                st.success(
                    f"ResNet50 Prediction: {label} ({conf:.4f})"
                )

            else:
                cnn_result = predict_image(
                    image,
                    cnn_model,
                    cnn_transform
                )

                resnet_result = predict_image(
                    image,
                    resnet_model,
                    resnet_transform
                )

                results = pd.DataFrame({
                    "Model": [
                        "CNN",
                        "ResNet50"
                    ],
                    "Prediction": [
                        cnn_result[0],
                        resnet_result[0]
                    ],
                    "Confidence": [
                        cnn_result[1],
                        resnet_result[1]
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True
                )

# ==========================================
# MODEL COMPARISON PAGE
# ==========================================

elif page == "Model Comparison":
    st.header("📊 Model Comparison")

    st.subheader("Text Model Metrics")

    try:
        text_metrics = pd.read_csv(
            "results/metrics.csv"
        )

        text_metrics = text_metrics[
            text_metrics["Model"].isin(
                [
                    "BiLSTM",
                    "BERT"
                ]
            )
        ]

        st.dataframe(
            text_metrics,
            use_container_width=True
        )

        st.bar_chart(
            text_metrics.set_index("Model")["Accuracy"]
        )

    except Exception:
        st.info("Text metrics file not found. Run src/text/evaluate.py first.")

    st.subheader("Image Model Metrics")

    try:
        image_metrics = pd.read_csv(
            "results/image_metrics.csv"
        )

        st.dataframe(
            image_metrics,
            use_container_width=True
        )

        st.bar_chart(
            image_metrics.set_index("Model")["Accuracy"]
        )

    except Exception:
        st.info("Image metrics file not found. Run src/image/evaluate.py first.")