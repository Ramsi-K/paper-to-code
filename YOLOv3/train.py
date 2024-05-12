import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv3
from dataset import YOLODataset, download_voc_dataset
from loss import YOLOLoss
from tqdm import tqdm  # Import tqdm for progress bar
from config import (
    DEVICE,
    BATCH_SIZE,
    IMG_DIR,
    LABEL_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    num_workers,
    PASCAL_CLASSES,
    train_transforms,
)


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)  # Stack images for batching
    return images, targets  # Return targets as list to avoid collate issues


def train_fn(data_loader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, leave=True)
    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(DEVICE)
        preds = model(images)
        loss = 0
        for target in targets:
            for i, pred in enumerate(preds):
                target[i] = target[i].to(DEVICE)
                loss += loss_fn(pred, target[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # Return average loss for the epoch
    return total_loss / len(data_loader)


def main():
    download_voc_dataset(year="2012", root="data/VOC")

    # Initialize dataset and DataLoader
    train_dataset = YOLODataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=[13, 26, 52],
        anchors=[
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ],
        C=len(PASCAL_CLASSES),
        transform=train_transforms,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize model, optimizer, and loss function
    model = YOLOv3(num_classes=len(PASCAL_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YOLOLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Average Loss: {avg_loss:.4f}")

        # Save model checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"yolov3_epoch{epoch + 1}.pth")
            print(f"Model checkpoint saved at epoch {epoch + 1}")


if __name__ == "__main__":
    main()
