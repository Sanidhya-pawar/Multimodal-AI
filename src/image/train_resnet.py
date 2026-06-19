import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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

print("\nClasses:")
print(train_dataset.classes)

print("\nTotal Images:")
print(len(train_dataset))

# ==========================================
# LOAD PRETRAINED RESNET50
# ==========================================

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Replace final layer

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    6
)

model = model.to(DEVICE)

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("\nTraining ResNet50...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

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
    "saved_models/resnet50.pth"
)

print(
    "\nModel saved to saved_models/resnet50.pth"
)