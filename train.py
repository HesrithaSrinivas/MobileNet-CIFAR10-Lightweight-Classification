"""
Final Year Project
MobileNetV2 on CIFAR-10 for Lightweight Image Classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR10("data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        ce = self.ce(outputs, targets)
        probs = torch.softmax(outputs, dim=1)
        penalty = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
        return ce + 0.05 * penalty

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

for p in model.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model = model.to(DEVICE)

criterion = CustomLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

history = {"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    correct,total,loss_sum = 0,0,0

    for images,labels in train_loader:
        images,labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()*images.size(0)
        preds = outputs.argmax(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    train_acc = correct/total
    train_loss = loss_sum/total

    model.eval()
    correct,total,loss_sum = 0,0,0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_sum += loss.item()*images.size(0)

            preds = outputs.argmax(1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct/total
    val_loss = loss_sum/total

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),"models/best_mobilenetv2.pth")

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_acc:.4f}")

plt.figure()
plt.plot(history["train_acc"])
plt.plot(history["val_acc"])
plt.legend(["Train","Validation"])
plt.savefig("outputs/training_curves.png")

report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
with open("outputs/classification_report.txt","w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("outputs/confusion_matrix.png")

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
