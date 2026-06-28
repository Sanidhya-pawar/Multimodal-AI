import os
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForSequenceClassification

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

MODEL_NAME = "facebook/wav2vec2-base"

SAMPLE_RATE = 16000
DURATION = 1

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5

# Use this to reduce training time.
# Set to None if you want full dataset.
MAX_FILES_PER_CLASS = 500

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

    wav_files = [
        file_name
        for file_name in os.listdir(cls_folder)
        if file_name.endswith(".wav")
    ]

    random.seed(42)
    random.shuffle(wav_files)

    if MAX_FILES_PER_CLASS is not None:
        wav_files = wav_files[:MAX_FILES_PER_CLASS]

    for file_name in wav_files:

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
# FEATURE EXTRACTOR
# ==========================================

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME
)

# ==========================================
# DATASET CLASS
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

            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        else:

            audio = audio[:target_length]

        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.squeeze(0)

        return (
            input_values,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

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
# MODEL
# ==========================================

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(SELECTED_CLASSES)
)

# Freeze feature extractor to reduce training time and VRAM usage
model.wav2vec2.feature_extractor._freeze_parameters()

model.to(DEVICE)

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

# ==========================================
# TRAINING
# ==========================================

print("\nTraining Wav2Vec2...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for input_values, labels in train_loader:

        input_values = input_values.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_values=input_values
        )

        loss = criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

save_path = "saved_models/audio_wav2vec2"

os.makedirs(
    save_path,
    exist_ok=True
)

model.save_pretrained(
    save_path
)

feature_extractor.save_pretrained(
    save_path
)

torch.save(
    {
        "classes": SELECTED_CLASSES,
        "class_to_idx": class_to_idx
    },
    os.path.join(
        save_path,
        "label_info.pth"
    )
)

print(
    "\nModel saved to saved_models/audio_wav2vec2"
)