import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torchvision import models

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "datasets/video/Real Life Violence Dataset"

CLASSES = [
    "NonViolence",
    "Violence"
]

NUM_FRAMES = 12
IMAGE_SIZE = 112
BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# CREATE VIDEO LIST
# ==========================================

video_paths = []
labels = []

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(CLASSES)
}

for class_name in CLASSES:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for file_name in os.listdir(class_path):

        if file_name.endswith((".mp4", ".avi")):

            video_paths.append(
                os.path.join(class_path, file_name)
            )

            labels.append(
                class_to_idx[class_name]
            )

_, test_paths, _, test_labels = train_test_split(
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==========================================
# FRAME LOADER
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
            IMAGE_SIZE,
            IMAGE_SIZE
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
                    IMAGE_SIZE,
                    IMAGE_SIZE,
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
                    IMAGE_SIZE,
                    IMAGE_SIZE
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

# ==========================================
# DATASET
# ==========================================

class VideoDataset(Dataset):

    def __init__(
        self,
        video_paths,
        labels
    ):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(
        self,
        idx
    ):
        frames = load_video_frames(
            self.video_paths[idx]
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return frames, label

test_dataset = VideoDataset(
    test_paths,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# CNN + LSTM MODEL
# ==========================================

class CNNLSTM(nn.Module):

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

        self.fc = nn.Linear(
            128,
            len(CLASSES)
        )

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

        output = self.fc(
            hidden[-1]
        )

        return output

# ==========================================
# MOBILENET + LSTM MODEL
# ==========================================

class MobileNetLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        mobilenet = models.mobilenet_v2(
            weights=None
        )

        self.feature_extractor = mobilenet.features

        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(
            128,
            len(CLASSES)
        )

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

        output = self.fc(
            hidden[-1]
        )

        return output

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

# ==========================================
# EVALUATE FUNCTION
# ==========================================

def evaluate_model(
    model,
    model_path,
    model_name
):

    print(f"\nEvaluating {model_name}...")

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    start = time.time()

    with torch.no_grad():

        for frames, labels in test_loader:

            frames = frames.to(DEVICE)

            outputs = model(frames)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    end = time.time()

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["Model"] = model_name
    metrics["Inference_Time"] = end - start

    return metrics

# ==========================================
# RUN EVALUATION
# ==========================================

results = []

results.append(
    evaluate_model(
        CNNLSTM(),
        "saved_models/video_cnn_lstm.pth",
        "CNN + LSTM"
    )
)

results.append(
    evaluate_model(
        MobileNetLSTM(),
        "saved_models/video_mobilenet_lstm.pth",
        "MobileNetV2 + LSTM"
    )
)

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

print("\nVideo Benchmark Results\n")
print(results_df)

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/video_metrics.csv",
    index=False
)

print(
    "\nSaved to results/video_metrics.csv"
)