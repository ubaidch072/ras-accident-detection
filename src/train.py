import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ================== CONFIG ==================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 5
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== DATA LOADING ==================

def get_dataloaders():
    print("Using device:", device)

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    # save class names to a file for inference
    with open(PROJECT_ROOT / "classes.txt", "w") as f:
        for c in class_names:
            f.write(c + "\n")

    # IMPORTANT: num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_names


# ================== MODEL ==================

def build_model(num_classes: int):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model.to(device)


# ================== ONE EPOCH ==================

def run_epoch(loader, model, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if optimizer:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / total
    acc = correct / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


# ================== MAIN TRAINING LOOP ==================

def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    model = build_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc, _, _ = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_acc, val_labels, val_preds = run_epoch(val_loader, model, criterion, optimizer=None)

        elapsed = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  (time: {elapsed:.1f}s)")

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), PROJECT_ROOT / "best_model.pt")
            print("  -> New best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Final test evaluation
    test_loss, test_acc, test_labels, test_preds = run_epoch(test_loader, model, criterion, optimizer=None)
    print("\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    main()
