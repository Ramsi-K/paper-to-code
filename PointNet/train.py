# train.py

import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import PointNet, PointNetSegmentation
import config
import torch.optim as optim
import torch.nn.functional as F


def train_classification():
    # Load dataset
    train_dataset = CustomDataset(
        root=config.DATA_DIR, split="train", task="classification"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize model
    model = PointNet(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            pred, trans_feat = model(points)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}"
        )

    print("Training completed.")


def train_segmentation():
    # Load dataset
    train_dataset = CustomDataset(
        root=config.DATA_DIR, split="train", task="segmentation"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize model
    model = PointNetSegmentation(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (points, seg_labels) in enumerate(train_loader):
            points, seg_labels = points.to(config.DEVICE), seg_labels.to(
                config.DEVICE
            )

            optimizer.zero_grad()
            pred, trans_feat = model(points)
            loss = F.cross_entropy(pred, seg_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}"
        )

    print("Training completed.")


if __name__ == "__main__":
    train_classification()
    train_segmentation()
