import os
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "datasets/audio/speech_commands"

SELECTED_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "stop",
    "go"
]

SAMPLE_RATE = 16000
DURATION = 1
N_MELS = 64
MAX_TIME_STEPS = 64

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# CREATE FILE LIST
# ==========================================

file_paths = []
labels = []

class_to_idx = {
    cls_name: idx
    for idx, cls_name in enumerate(SELECTED_CLASSES)
}

for cls_name in SELECTED_CLASSES:

    cls_folder = os.path.join(
        DATASET_PATH,
        cls_name
    )

    for file_name in os.listdir(cls_folder):

        if file_name.endswith(".wav"):

            file_paths.append(
                os.path.join(cls_folder, file_name)
            )

            labels.append(
                class_to_idx[cls_name]
            )

print("\nSelected Classes:")
print(SELECTED_CLASSES)

print("\nTotal Audio Files:")
print(len(file_paths))

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==========================================
# AUDIO DATASET
# ==========================================

class SpeechCommandDataset(Dataset):

    def __init__(
        self,
        file_paths,
        labels
    ):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(
        self,
        idx
    ):
        audio_path = self.file_paths[idx]

        label = self.labels[idx]

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE
        )

        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:

            padding = target_length - len(audio)

            audio = np.pad(
                audio,
                (0, padding)
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
                (
                    (0, 0),
                    (0, pad_width)
                )
            )

        else:

            mel_db = mel_db[:, :MAX_TIME_STEPS]

        mel_db = torch.tensor(
            mel_db,
            dtype=torch.float32
        )

        mel_db = mel_db.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel_db, label

# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = SpeechCommandDataset(
    train_paths,
    train_labels
)

test_dataset = SpeechCommandDataset(
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
# CNN MODEL
# ==========================================

class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            64 * 16 * 16,
            128
        )

        self.fc2 = nn.Linear(
            128,
            len(SELECTED_CLASSES)
        )

    def forward(
        self,
        x
    ):

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
# MODEL SETUP
# ==========================================

model = AudioCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("\nTraining Audio CNN...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

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
        "classes": SELECTED_CLASSES,
        "class_to_idx": class_to_idx
    },
    "saved_models/audio_cnn.pth"
)

print(
    "\nModel saved to saved_models/audio_cnn.pth"
)