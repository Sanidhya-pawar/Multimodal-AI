import os
import time
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "datasets/audio/speech_commands"

SELECTED_CLASSES = [
    "yes", "no", "up", "down",
    "left", "right", "stop", "go"
]

SAMPLE_RATE = 16000
DURATION = 1
N_MELS = 64
MAX_TIME_STEPS = 64

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# FILE LIST
# ==========================================

file_paths = []
labels = []

class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(SELECTED_CLASSES)
}

for cls_name in SELECTED_CLASSES:
    cls_folder = os.path.join(DATASET_PATH, cls_name)

    for file_name in os.listdir(cls_folder):
        if file_name.endswith(".wav"):
            file_paths.append(os.path.join(cls_folder, file_name))
            labels.append(class_to_idx[cls_name])

_, test_paths, _, test_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==========================================
# AUDIO CNN DATASET
# ==========================================

class AudioCNNDataset(Dataset):

    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

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

        mel_db = torch.tensor(
            mel_db,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel_db, label

# ==========================================
# WAV2VEC2 DATASET
# ==========================================

class Wav2VecDataset(Dataset):

    def __init__(self, file_paths, labels, feature_extractor):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

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

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.squeeze(0)

        return input_values, torch.tensor(label, dtype=torch.long)

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
        self.fc2 = nn.Linear(128, len(SELECTED_CLASSES))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted")
    }

results = []

# ==========================================
# EVALUATE AUDIO CNN
# ==========================================

print("\nEvaluating Audio CNN...")

cnn_dataset = AudioCNNDataset(
    test_paths,
    test_labels
)

cnn_loader = DataLoader(
    cnn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

cnn = AudioCNN()

checkpoint = torch.load(
    "saved_models/audio_cnn.pth",
    map_location=DEVICE
)

cnn.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn.to(DEVICE)
cnn.eval()

y_true = []
y_pred = []

start = time.time()

with torch.no_grad():
    for features, labels in cnn_loader:
        features = features.to(DEVICE)

        outputs = cnn(features)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

end = time.time()

metrics = calculate_metrics(y_true, y_pred)
metrics["Model"] = "Audio CNN"
metrics["Inference_Time"] = end - start

results.append(metrics)

# ==========================================
# EVALUATE WAV2VEC2
# ==========================================

print("\nEvaluating Wav2Vec2...")

wav2vec_path = "saved_models/audio_wav2vec2"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    wav2vec_path
)

wav2vec_dataset = Wav2VecDataset(
    test_paths,
    test_labels,
    feature_extractor
)

wav2vec_loader = DataLoader(
    wav2vec_dataset,
    batch_size=4,
    shuffle=False
)

wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    wav2vec_path
)

wav2vec_model.to(DEVICE)
wav2vec_model.eval()

y_true = []
y_pred = []

start = time.time()

with torch.no_grad():
    for input_values, labels in wav2vec_loader:
        input_values = input_values.to(DEVICE)

        outputs = wav2vec_model(
            input_values=input_values
        )

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

end = time.time()

metrics = calculate_metrics(y_true, y_pred)
metrics["Model"] = "Wav2Vec2"
metrics["Inference_Time"] = end - start

results.append(metrics)

# ==========================================
# SAVE RESULTS
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

print("\nAudio Benchmark Results\n")
print(results_df)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/audio_metrics.csv",
    index=False
)

print("\nSaved to results/audio_metrics.csv")