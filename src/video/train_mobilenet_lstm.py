import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
EPOCHS = 5
LEARNING_RATE = 0.0001

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

print("\nTotal Videos:")
print(len(video_paths))

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_paths, test_paths, train_labels, test_labels = train_test_split(
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTrain Videos:", len(train_paths))
print("Test Videos :", len(test_paths))

# ==========================================
# VIDEO FRAME LOADER
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
        video_path = self.video_paths[idx]

        label = self.labels[idx]

        frames = load_video_frames(
            video_path
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return frames, label

# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = VideoDataset(
    train_paths,
    train_labels
)

test_dataset = VideoDataset(
    test_paths,
    test_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# MOBILENET + LSTM MODEL
# ==========================================

class MobileNetLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        self.feature_extractor = mobilenet.features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

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

    def forward(
        self,
        x
    ):
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
# MODEL SETUP
# ==========================================

model = MobileNetLSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("\nTraining MobileNetV2 + LSTM...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for frames, labels in train_loader:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "class_to_idx": class_to_idx,
        "num_frames": NUM_FRAMES,
        "image_size": IMAGE_SIZE
    },
    "saved_models/video_mobilenet_lstm.pth"
)

print(
    "\nModel saved to saved_models/video_mobilenet_lstm.pth"
)