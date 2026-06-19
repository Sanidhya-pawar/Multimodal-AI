import os
import time
import torch
import torch.nn as nn
import pandas as pd

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# CONFIG
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32

# ==========================================
# TRANSFORMS
# ==========================================

cnn_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================================
# TEST DATASETS
# ==========================================

cnn_test_dataset = datasets.ImageFolder(
    root="datasets/image/seg_test/seg_test",
    transform=cnn_transform
)

resnet_test_dataset = datasets.ImageFolder(
    root="datasets/image/seg_test/seg_test",
    transform=resnet_transform
)

cnn_test_loader = DataLoader(
    cnn_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

resnet_test_loader = DataLoader(
    resnet_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# CNN MODEL
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
# METRICS FUNCTION
# ==========================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "Accuracy":
            accuracy_score(y_true, y_pred),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                average="weighted"
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                average="weighted"
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                average="weighted"
            )
    }

# ==========================================
# RESULTS
# ==========================================

results = []

# ==========================================
# CNN EVALUATION
# ==========================================

print("\nEvaluating CNN...")

cnn = CNN()

cnn.load_state_dict(
    torch.load(
        "saved_models/image_cnn.pth",
        map_location=DEVICE
    )
)

cnn.to(DEVICE)

cnn.eval()

y_true = []
y_pred = []

start = time.time()

with torch.no_grad():

    for images, labels in cnn_test_loader:

        images = images.to(DEVICE)

        outputs = cnn(images)

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

metrics["Model"] = "CNN"

metrics["Inference_Time"] = (
    end - start
)

results.append(metrics)

# ==========================================
# RESNET50 EVALUATION
# ==========================================

print("\nEvaluating ResNet50...")

resnet = models.resnet50()

resnet.fc = nn.Linear(
    resnet.fc.in_features,
    6
)

resnet.load_state_dict(
    torch.load(
        "saved_models/resnet50.pth",
        map_location=DEVICE
    )
)

resnet.to(DEVICE)

resnet.eval()

y_true = []
y_pred = []

start = time.time()

with torch.no_grad():

    for images, labels in resnet_test_loader:

        images = images.to(DEVICE)

        outputs = resnet(images)

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

metrics["Model"] = "ResNet50"

metrics["Inference_Time"] = (
    end - start
)

results.append(metrics)

# ==========================================
# RESULTS DATAFRAME
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

print("\nImage Benchmark Results\n")

print(results_df)

# ==========================================
# SAVE CSV
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/image_metrics.csv",
    index=False
)

print(
    "\nSaved to results/image_metrics.csv"
)