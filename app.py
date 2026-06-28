import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
import joblib
import cv2

from PIL import Image
from collections import Counter
from torchvision import transforms, models

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

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
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

# ==========================================
# AUDIO CONFIG
# ==========================================

AUDIO_CLASSES = [
    "yes", "no", "up", "down",
    "left", "right", "stop", "go"
]

SAMPLE_RATE = 16000
DURATION = 1
N_MELS = 64
MAX_TIME_STEPS = 64

# ==========================================
# TABULAR CONFIG
# ==========================================

TABULAR_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# ==========================================
# VIDEO CONFIG
# ==========================================

VIDEO_CLASSES = [
    "NonViolence",
    "Violence"
]

NUM_FRAMES = 12
VIDEO_IMAGE_SIZE = 112

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
# TEXT MODEL
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
            (hidden[-2], hidden[-1]),
            dim=1
        )

        x = self.fc(hidden)

        return self.sigmoid(x)

# ==========================================
# IMAGE CNN MODEL
# ==========================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==========================================
# AUDIO CNN MODEL
# ==========================================

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, len(AUDIO_CLASSES))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==========================================
# VIDEO CNN + LSTM MODEL
# ==========================================

class VideoCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.feature_size = 128 * 14 * 14

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, len(VIDEO_CLASSES))

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        x = x.view(
            batch_size * seq_len,
            c,
            h,
            w
        )

        features = self.cnn(x)

        features = features.view(
            batch_size,
            seq_len,
            -1
        )

        _, (hidden, _) = self.lstm(features)

        output = self.fc(hidden[-1])

        return output

# ==========================================
# VIDEO MOBILENET + LSTM MODEL
# ==========================================

class VideoMobileNetLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=None)

        self.feature_extractor = mobilenet.features

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, len(VIDEO_CLASSES))

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        x = x.view(
            batch_size * seq_len,
            c,
            h,
            w
        )

        features = self.feature_extractor(x)

        features = self.avg_pool(features)

        features = features.view(
            batch_size,
            seq_len,
            -1
        )

        _, (hidden, _) = self.lstm(features)

        output = self.fc(hidden[-1])

        return output

# ==========================================
# LOAD ALL MODELS
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
    bilstm.to(DEVICE)
    bilstm.eval()

    bert_tokenizer = AutoTokenizer.from_pretrained(
        "saved_models/bert"
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_models/bert"
    )
    bert_model.to(DEVICE)
    bert_model.eval()

    image_cnn = CNN()
    image_cnn.load_state_dict(
        torch.load(
            "saved_models/image_cnn.pth",
            map_location=DEVICE
        )
    )
    image_cnn.to(DEVICE)
    image_cnn.eval()

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

    audio_cnn = AudioCNN()
    audio_checkpoint = torch.load(
        "saved_models/audio_cnn.pth",
        map_location=DEVICE
    )
    audio_cnn.load_state_dict(
        audio_checkpoint["model_state_dict"]
    )
    audio_cnn.to(DEVICE)
    audio_cnn.eval()

    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "saved_models/audio_wav2vec2"
    )

    wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "saved_models/audio_wav2vec2"
    )
    wav2vec_model.to(DEVICE)
    wav2vec_model.eval()

    random_forest = joblib.load(
        "saved_models/random_forest.pkl"
    )

    xgboost_model = joblib.load(
        "saved_models/xgboost.pkl"
    )

    catboost_model = joblib.load(
        "saved_models/catboost.pkl"
    )

    video_cnn_lstm = VideoCNNLSTM()
    video_cnn_checkpoint = torch.load(
        "saved_models/video_cnn_lstm.pth",
        map_location=DEVICE
    )
    video_cnn_lstm.load_state_dict(
        video_cnn_checkpoint["model_state_dict"]
    )
    video_cnn_lstm.to(DEVICE)
    video_cnn_lstm.eval()

    video_mobilenet_lstm = VideoMobileNetLSTM()
    video_mobile_checkpoint = torch.load(
        "saved_models/video_mobilenet_lstm.pth",
        map_location=DEVICE
    )
    video_mobilenet_lstm.load_state_dict(
        video_mobile_checkpoint["model_state_dict"]
    )
    video_mobilenet_lstm.to(DEVICE)
    video_mobilenet_lstm.eval()

    return (
        bilstm,
        bert_tokenizer,
        bert_model,
        image_cnn,
        resnet_model,
        audio_cnn,
        wav2vec_extractor,
        wav2vec_model,
        random_forest,
        xgboost_model,
        catboost_model,
        video_cnn_lstm,
        video_mobilenet_lstm
    )


(
    bilstm,
    bert_tokenizer,
    bert_model,
    image_cnn,
    resnet_model,
    audio_cnn,
    wav2vec_extractor,
    wav2vec_model,
    random_forest,
    xgboost_model,
    catboost_model,
    video_cnn_lstm,
    video_mobilenet_lstm
) = load_models()

# ==========================================
# TEXT PREDICTION
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
# IMAGE PREDICTION
# ==========================================

image_cnn_transform = transforms.Compose([
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

    label = IMAGE_CLASSES[pred.item()]

    return label, confidence.item()

# ==========================================
# AUDIO PREDICTION
# ==========================================

def load_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    return audio


def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    if mel_db.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width))
        )
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]

    mel_tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    return mel_tensor


def predict_audio_cnn(audio_path):
    audio = load_audio(audio_path)

    mel_tensor = audio_to_mel(audio).to(DEVICE)

    with torch.no_grad():
        outputs = audio_cnn(mel_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    label = AUDIO_CLASSES[pred.item()]

    return label, confidence.item()


def predict_wav2vec2(audio_path):
    audio = load_audio(audio_path)

    inputs = wav2vec_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = wav2vec_model(
            input_values=input_values
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    label = AUDIO_CLASSES[pred.item()]

    return label, confidence.item()

# ==========================================
# TABULAR PREDICTION
# ==========================================

def predict_tabular(model, data):
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
# VIDEO PREDICTION
# ==========================================

def load_video_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()

        return torch.zeros(
            NUM_FRAMES,
            3,
            VIDEO_IMAGE_SIZE,
            VIDEO_IMAGE_SIZE
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    frames = []

    for frame_idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        success, frame = cap.read()

        if not success:
            frame = np.zeros(
                (
                    VIDEO_IMAGE_SIZE,
                    VIDEO_IMAGE_SIZE,
                    3
                ),
                dtype=np.uint8
            )
        else:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (
                    VIDEO_IMAGE_SIZE,
                    VIDEO_IMAGE_SIZE
                )
            )

        frame = frame / 255.0

        frame = np.transpose(
            frame,
            (
                2,
                0,
                1
            )
        )

        frames.append(frame)

    cap.release()

    frames = np.array(
        frames,
        dtype=np.float32
    )

    return torch.tensor(
        frames,
        dtype=torch.float32
    )


def predict_video(model, video_path):
    frames = load_video_frames(video_path)

    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(frames)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    label = VIDEO_CLASSES[pred.item()]

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
        "Audio Classification",
        "Heart Disease Prediction",
        "Video Violence Detection",
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
        This platform compares machine learning and deep learning models
        across multiple data modalities: Text, Image, Audio, Tabular, and Video.
        """
    )

    st.subheader("Implemented Modules")

    st.write("✅ Text Module: BiLSTM and BERT")
    st.write("✅ Image Module: CNN and ResNet50")
    st.write("✅ Audio Module: Audio CNN and Wav2Vec2")
    st.write("✅ Tabular Module: Random Forest, XGBoost, and CatBoost")
    st.write("✅ Video Module: CNN+LSTM and MobileNetV2+LSTM")
    st.write("✅ Evaluation Metrics and Model Comparison")

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
                    "Model": ["BiLSTM", "BERT"],
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
                    image_cnn,
                    image_cnn_transform
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
                    image_cnn,
                    image_cnn_transform
                )

                resnet_result = predict_image(
                    image,
                    resnet_model,
                    resnet_transform
                )

                results = pd.DataFrame({
                    "Model": ["CNN", "ResNet50"],
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
# AUDIO PAGE
# ==========================================

elif page == "Audio Classification":
    st.header("🎤 Audio Command Classification")

    uploaded_audio = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    model_choice = st.selectbox(
        "Select Audio Model",
        [
            "Audio CNN",
            "Wav2Vec2",
            "Compare Both"
        ]
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("Predict Audio Command"):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp_audio:
                temp_audio.write(uploaded_audio.getvalue())
                temp_audio_path = temp_audio.name

            if model_choice == "Audio CNN":
                label, conf = predict_audio_cnn(
                    temp_audio_path
                )

                st.success(
                    f"Audio CNN Prediction: {label} ({conf:.4f})"
                )

            elif model_choice == "Wav2Vec2":
                label, conf = predict_wav2vec2(
                    temp_audio_path
                )

                st.success(
                    f"Wav2Vec2 Prediction: {label} ({conf:.4f})"
                )

            else:
                cnn_result = predict_audio_cnn(
                    temp_audio_path
                )

                wav_result = predict_wav2vec2(
                    temp_audio_path
                )

                results = pd.DataFrame({
                    "Model": ["Audio CNN", "Wav2Vec2"],
                    "Prediction": [
                        cnn_result[0],
                        wav_result[0]
                    ],
                    "Confidence": [
                        cnn_result[1],
                        wav_result[1]
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True
                )

# ==========================================
# TABULAR PAGE
# ==========================================

elif page == "Heart Disease Prediction":
    st.header("❤️ Heart Disease Prediction")

    st.warning(
        "This is a machine learning demo project and should not be used as medical advice."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 52)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.number_input("Chest Pain Type (cp)", 0, 3, 0)
        trestbps = st.number_input("Resting BP", 80, 250, 125)
        chol = st.number_input("Cholesterol", 100, 600, 212)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.number_input("Rest ECG", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate", 60, 250, 168)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])

    with col3:
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.number_input("Slope", 0, 2, 2)
        ca = st.number_input("CA", 0, 4, 2)
        thal = st.number_input("Thal", 0, 3, 3)

    model_choice = st.selectbox(
        "Select Tabular Model",
        [
            "Random Forest",
            "XGBoost",
            "CatBoost",
            "Compare All"
        ]
    )

    patient_data = pd.DataFrame(
        [[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]],
        columns=TABULAR_FEATURES
    )

    if st.button("Predict Heart Disease"):
        if model_choice == "Random Forest":
            label, conf = predict_tabular(
                random_forest,
                patient_data
            )

            st.success(
                f"Random Forest Prediction: {label} ({conf:.4f})"
            )

        elif model_choice == "XGBoost":
            label, conf = predict_tabular(
                xgboost_model,
                patient_data
            )

            st.success(
                f"XGBoost Prediction: {label} ({conf:.4f})"
            )

        elif model_choice == "CatBoost":
            label, conf = predict_tabular(
                catboost_model,
                patient_data
            )

            st.success(
                f"CatBoost Prediction: {label} ({conf:.4f})"
            )

        else:
            rf_result = predict_tabular(
                random_forest,
                patient_data
            )

            xgb_result = predict_tabular(
                xgboost_model,
                patient_data
            )

            cat_result = predict_tabular(
                catboost_model,
                patient_data
            )

            results = pd.DataFrame({
                "Model": [
                    "Random Forest",
                    "XGBoost",
                    "CatBoost"
                ],
                "Prediction": [
                    rf_result[0],
                    xgb_result[0],
                    cat_result[0]
                ],
                "Confidence": [
                    rf_result[1],
                    xgb_result[1],
                    cat_result[1]
                ]
            })

            st.dataframe(
                results,
                use_container_width=True
            )

# ==========================================
# VIDEO PAGE
# ==========================================

elif page == "Video Violence Detection":
    st.header("🎥 Video Violence Detection")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=[
            "mp4",
            "avi"
        ]
    )

    model_choice = st.selectbox(
        "Select Video Model",
        [
            "CNN + LSTM",
            "MobileNetV2 + LSTM",
            "Compare Both"
        ]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Predict Video"):
            suffix = ".mp4"

            if uploaded_video.name.lower().endswith(".avi"):
                suffix = ".avi"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_video:
                temp_video.write(uploaded_video.getvalue())
                temp_video_path = temp_video.name

            if model_choice == "CNN + LSTM":
                label, conf = predict_video(
                    video_cnn_lstm,
                    temp_video_path
                )

                st.success(
                    f"CNN + LSTM Prediction: {label} ({conf:.4f})"
                )

            elif model_choice == "MobileNetV2 + LSTM":
                label, conf = predict_video(
                    video_mobilenet_lstm,
                    temp_video_path
                )

                st.success(
                    f"MobileNetV2 + LSTM Prediction: {label} ({conf:.4f})"
                )

            else:
                cnn_result = predict_video(
                    video_cnn_lstm,
                    temp_video_path
                )

                mobile_result = predict_video(
                    video_mobilenet_lstm,
                    temp_video_path
                )

                results = pd.DataFrame({
                    "Model": [
                        "CNN + LSTM",
                        "MobileNetV2 + LSTM"
                    ],
                    "Prediction": [
                        cnn_result[0],
                        mobile_result[0]
                    ],
                    "Confidence": [
                        cnn_result[1],
                        mobile_result[1]
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True
                )

# ==========================================
# MODEL COMPARISON
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

    st.subheader("Audio Model Metrics")

    try:
        audio_metrics = pd.read_csv(
            "results/audio_metrics.csv"
        )

        st.dataframe(
            audio_metrics,
            use_container_width=True
        )

        st.bar_chart(
            audio_metrics.set_index("Model")["Accuracy"]
        )

    except Exception:
        st.info("Audio metrics file not found. Run src/audio/evaluate.py first.")

    st.subheader("Tabular Model Metrics")

    try:
        tabular_metrics = pd.read_csv(
            "results/tabular_metrics.csv"
        )

        st.dataframe(
            tabular_metrics,
            use_container_width=True
        )

        st.bar_chart(
            tabular_metrics.set_index("Model")["Accuracy"]
        )

    except Exception:
        st.info("Tabular metrics file not found. Run src/tabular/evaluate.py first.")

    st.subheader("Video Model Metrics")

    try:
        video_metrics = pd.read_csv(
            "results/video_metrics.csv"
        )

        st.dataframe(
            video_metrics,
            use_container_width=True
        )

        st.bar_chart(
            video_metrics.set_index("Model")["Accuracy"]
        )

    except Exception:
        st.info("Video metrics file not found. Run src/video/evaluate.py first.")