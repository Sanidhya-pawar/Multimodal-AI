# src/image/inference.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# =========================
# CNN MODEL
# =========================

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


# =========================
# LOAD CNN
# =========================

cnn_model = CNN().to(DEVICE)

cnn_model.load_state_dict(
    torch.load(
        "saved_models/image_cnn.pth",
        map_location=DEVICE
    )
)

cnn_model.eval()


# =========================
# LOAD RESNET50
# =========================

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

resnet_model = resnet_model.to(DEVICE)

resnet_model.eval()


# =========================
# TRANSFORMS
# =========================

cnn_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# PREDICTION FUNCTION
# =========================

def predict_image(
    image_path,
    model,
    transform
):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASSES[
        predicted.item()
    ]

    return predicted_class, confidence.item()


# =========================
# MAIN
# =========================

print("=" * 50)
print("IMAGE CLASSIFICATION INFERENCE")
print("=" * 50)

while True:
    image_path = input(
        "\nEnter image path or 'quit': "
    )

    if image_path.lower() == "quit":
        break

    cnn_pred, cnn_conf = predict_image(
        image_path,
        cnn_model,
        cnn_transform
    )

    resnet_pred, resnet_conf = predict_image(
        image_path,
        resnet_model,
        resnet_transform
    )

    print("\nResults")
    print("-" * 50)

    print(
        f"CNN      : {cnn_pred} "
        f"({cnn_conf:.4f})"
    )

    print(
        f"ResNet50 : {resnet_pred} "
        f"({resnet_conf:.4f})"
    )