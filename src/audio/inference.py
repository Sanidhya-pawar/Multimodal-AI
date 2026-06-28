import librosa
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

# ==============================
# CONFIG
# ==============================

SAMPLE_RATE = 16000
DURATION = 1
N_MELS = 64
MAX_TIME_STEPS = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "yes", "no", "up", "down",
    "left", "right", "stop", "go"
]

# ==============================
# AUDIO CNN MODEL
# ==============================

class AudioCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==============================
# AUDIO HELPERS
# ==============================

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

# ==============================
# LOAD MODELS
# ==============================

cnn_model = AudioCNN().to(DEVICE)

cnn_checkpoint = torch.load(
    "saved_models/audio_cnn.pth",
    map_location=DEVICE
)

cnn_model.load_state_dict(
    cnn_checkpoint["model_state_dict"]
)

cnn_model.eval()

wav2vec_path = "saved_models/audio_wav2vec2"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    wav2vec_path
)

wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    wav2vec_path
)

wav2vec_model.to(DEVICE)
wav2vec_model.eval()

# ==============================
# PREDICTION FUNCTIONS
# ==============================

def predict_cnn(audio_path):
    audio = load_audio(audio_path)

    mel_tensor = audio_to_mel(audio).to(DEVICE)

    with torch.no_grad():
        outputs = cnn_model(mel_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    return CLASSES[pred.item()], confidence.item()


def predict_wav2vec2(audio_path):
    audio = load_audio(audio_path)

    inputs = feature_extractor(
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

    return CLASSES[pred.item()], confidence.item()

# ==============================
# MAIN LOOP
# ==============================

print("=" * 50)
print("AUDIO COMMAND CLASSIFICATION")
print("=" * 50)

while True:

    audio_path = input(
        "\nEnter audio file path or 'quit': "
    )

    if audio_path.lower() == "quit":
        break

    cnn_label, cnn_conf = predict_cnn(
        audio_path
    )

    wav_label, wav_conf = predict_wav2vec2(
        audio_path
    )

    print("\nResults")
    print("-" * 50)

    print(
        f"Audio CNN : {cnn_label} "
        f"({cnn_conf:.4f})"
    )

    print(
        f"Wav2Vec2  : {wav_label} "
        f"({wav_conf:.4f})"
    )