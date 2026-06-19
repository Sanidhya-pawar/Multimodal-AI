import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# CONFIG
# ==========================================

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# TRANSFORMS
# ==========================================

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# ==========================================
# DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    root="datasets/image/seg_train/seg_train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# MODEL
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
# MODEL SETUP
# ==========================================

model = CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("\nTraining CNN...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "saved_models/image_cnn.pth"
)

print(
    "\nModel saved to saved_models/image_cnn.pth"
)