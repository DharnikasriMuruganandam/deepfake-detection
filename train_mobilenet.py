import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

train_data = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

test_data = datasets.ImageFolder(
    "dataset/test",
    transform=test_transform
)

print("Classes:", train_data.classes)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32
)

model = models.mobilenet_v2(
    weights="IMAGENET1K_V1"
)

for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-5:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 2)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

epochs = 35

best_acc = 0

print("\n🚀 Training MobileNetV2...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"| Loss: {total_loss:.2f} "
        f"| Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            "mobilenet_model.pth"
        )

        print(f"🔥 Best MobileNet Saved ({best_acc:.2f}%)")

print("\nMobileNet Training Completed")
print(f" Best Accuracy: {best_acc:.2f}%")