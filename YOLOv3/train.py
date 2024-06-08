import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv3
from dataset import YOLODataset, download_voc_dataset
from loss import YOLOLoss
from tqdm import tqdm  # For progress bar
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
    # ANCHORS,
)

ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # Anchor boxes used in YOLOv3
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]


# Custom collate function to handle batches with varying target sizes
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)
    return images, targets


# Training function
def train_fn(data_loader, model, optimizer, loss_fn, anchors):
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, leave=True)

    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(DEVICE)
        formatted_targets = []

        # Format targets for different scales
        for i in range(len(anchors)):
            scale_targets = torch.stack(
                [target[i].to(DEVICE) for target in targets]
            )
            scale_targets[..., 5] = (
                scale_targets[..., 5].clamp(0, len(PASCAL_CLASSES) - 1).long()
            )
            formatted_targets.append(scale_targets)

        preds = model(images)
        loss = 0
        for i, pred in enumerate(preds):
            anchor_tensor = torch.tensor(anchors[i], device=DEVICE).reshape(
                1, len(anchors[i]), 1, 1, 2
            )
            pred_objectness = pred[..., 0:1]
            pred_box = pred[..., 1:5]
            pred_class_scores = pred[..., 5:]
            max_class_index = pred_class_scores.argmax(
                dim=-1, keepdim=True
            ).float()
            combined_pred = torch.cat(
                (pred_objectness, pred_box, max_class_index), dim=-1
            )
            loss += loss_fn(combined_pred, formatted_targets[i], anchor_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


# Main training function
def main():
    download_voc_dataset(year="2012", root="data/VOC")

    train_dataset = YOLODataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=[8, 16, 32],
        anchors=ANCHORS,
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

    model = YOLOv3(num_classes=len(PASCAL_CLASSES)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YOLOLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, ANCHORS)
        print(f"Average Loss: {avg_loss:.4f}")
        scheduler.step()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "avg_loss": avg_loss,
            }
            torch.save(checkpoint, f"yolov3_epoch{epoch + 1}.pth")
            print(f"Model checkpoint saved at epoch {epoch + 1}")


if __name__ == "__main__":
    main()
