from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs.cpu().numpy().ravel())
        all_labels.extend(labels.cpu().numpy().ravel())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    f1 = f1_score(all_labels, preds, zero_division=0)
    return total_loss / len(loader.dataset), f1, all_probs, all_labels