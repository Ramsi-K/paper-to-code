# eval.py

import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import PointNet, PointNetSegmentation
import config
import torch.nn.functional as F


def evaluate_classification():
    # Load dataset
    test_dataset = CustomDataset(
        root=config.DATA_DIR, split="test", task="classification"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize model
    model = PointNet(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(config.DEVICE), labels.to(config.DEVICE)

            pred, _ = model(points)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Classification accuracy on test set: {accuracy}%")


def evaluate_segmentation():
    # Load dataset
    test_dataset = CustomDataset(
        root=config.DATA_DIR, split="test", task="segmentation"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize model
    model = PointNetSegmentation(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for points, seg_labels in test_loader:
            points, seg_labels = points.to(config.DEVICE), seg_labels.to(
                config.DEVICE
            )

            pred, _ = model(points)
            loss = F.cross_entropy(pred, seg_labels)
            total_loss += loss.item()

    print(f"Segmentation loss on test set: {total_loss / len(test_loader)}")


if __name__ == "__main__":
    evaluate_classification()
    evaluate_segmentation()
