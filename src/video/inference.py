import cv2
import torch
import torch.nn as nn
import numpy as np

from torchvision import models

# ==========================================
# CONFIG
# ==========================================

CLASSES = [
    "NonViolence",
    "Violence"
]

NUM_FRAMES = 12
IMAGE_SIZE = 112

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

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
# LOAD MODELS
# ==========================================

cnn_lstm = CNNLSTM().to(DEVICE)

cnn_checkpoint = torch.load(
    "saved_models/video_cnn_lstm.pth",
    map_location=DEVICE
)

cnn_lstm.load_state_dict(
    cnn_checkpoint["model_state_dict"]
)

cnn_lstm.eval()

mobilenet_lstm = MobileNetLSTM().to(DEVICE)

mobilenet_checkpoint = torch.load(
    "saved_models/video_mobilenet_lstm.pth",
    map_location=DEVICE
)

mobilenet_lstm.load_state_dict(
    mobilenet_checkpoint["model_state_dict"]
)

mobilenet_lstm.eval()

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_video(model, video_path):

    frames = load_video_frames(
        video_path
    )

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

    label = CLASSES[
        pred.item()
    ]

    return label, confidence.item()

# ==========================================
# MAIN LOOP
# ==========================================

print("=" * 50)
print("VIDEO VIOLENCE DETECTION")
print("=" * 50)

while True:

    video_path = input(
        "\nEnter video path or 'quit': "
    )

    if video_path.lower() == "quit":
        break

    cnn_label, cnn_conf = predict_video(
        cnn_lstm,
        video_path
    )

    mobile_label, mobile_conf = predict_video(
        mobilenet_lstm,
        video_path
    )

    print("\nResults")
    print("-" * 50)

    print(
        f"CNN + LSTM          : {cnn_label} "
        f"({cnn_conf:.4f})"
    )

    print(
        f"MobileNetV2 + LSTM : {mobile_label} "
        f"({mobile_conf:.4f})"
    )