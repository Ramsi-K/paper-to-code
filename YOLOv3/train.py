# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.YOLOv3 import YOLOv3
from models.yolo_loss import YOLOLoss
from dataset import VOCDataset, download_voc_dataset
from tqdm import tqdm  # Import tqdm for progress tracking

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import config


# def train():
#     """
#     Training loop for YOLOv3 model.
#     """
#     device = torch.device(config.DEVICE)
#     model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=config.LEARNING_RATE,
#         weight_decay=config.WEIGHT_DECAY,
#     )

#     anchors = config.ANCHORS
#     criterion = YOLOLoss(anchors, config.NUM_CLASSES, config.IMAGE_SIZE)

#     # Check and download dataset if not already present
#     download_voc_dataset(year="2012", root="data/VOC")

#     train_dataset = VOCDataset(
#         img_dir=config.IMAGES_PATH,
#         label_dir=config.ANNOTATIONS_PATH,
#         img_size=config.IMAGE_SIZE,
#         transform=config.TRAIN_AUGMENT,
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY,
#     )

#     model.train()
#     for epoch in range(config.NUM_EPOCHS):
#         epoch_loss = 0
#         progress_bar = tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
#         )
#         for batch_idx, (images, targets) in progress_bar:
#             images = images.to(device).float()
#             targets = [target.to(device) for target in targets]

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = 0
#             # Compute loss for each scale
#             for i in range(3):
#                 loss += criterion(outputs[i], targets)  # , anchors[i]
#             loss.backward()
#             optimizer.step()

#             # Update progress bar and accumulate epoch loss
#             epoch_loss += loss.item()
#             progress_bar.set_postfix({"Batch Loss": loss.item()})

#         # Display average loss per epoch
#         print(
#             f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}],\
#              Average Loss: {epoch_loss / len(train_loader):.4f}"
#         )

#         # Save checkpoint
#         if config.SAVE_CHECKPOINT:
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(
#                     config.CHECKPOINTS_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
#                 ),
#             )


def train():
    device = torch.device(config.DEVICE)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = YOLOLoss(config.ANCHORS, config.NUM_CLASSES, config.IMAGE_SIZE)

    # Prepare dataset and dataloader
    download_voc_dataset(year="2012", root="data/VOC")
    train_dataset = VOCDataset(
        img_dir=config.IMAGES_PATH,
        label_dir=config.ANNOTATIONS_PATH,
        img_size=config.IMAGE_SIZE,
        transform=config.TRAIN_AUGMENT,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
        )

        for batch_idx, (images, targets) in progress_bar:
            images = images.to(device).float()

            # Check image dimensions
            print(
                f"Image batch shape: {images.shape}\
                (Expected: [batch_size, 3, 416, 416])"
            )

            # Verify target shapes for each scale
            for i, target in enumerate(targets):
                print(
                    f"Target[{i}] shape: {target.shape}\
                    (Expected: [batch_size, 3, S, S, {5 + config.NUM_CLASSES}])"
                )

            optimizer.zero_grad()
            outputs = model(images)

            # Check output shapes for each scale
            for i, output in enumerate(outputs):
                print(
                    f"Output[{i}] shape: {output.shape}\
                    (Expected: [batch_size, 3 * ({5 + config.NUM_CLASSES}), S, S])"
                )

            loss = 0
            # Compute loss for each scale with scale-specific targets
            for i in range(3):
                scale_output = outputs[i]
                scale_target = targets[i]

                # Check dimensions of data before forwarding to criterion
                print(
                    f"Forwarding to YOLOLoss: Output[{i}].shape\
                    = {scale_output.shape},\
                    Target[{i}].shape= {scale_target.shape}"
                )

                try:
                    loss += criterion(scale_output, scale_target)
                except RuntimeError as e:
                    print(f"Error in YOLOLoss at scale {i}: {e}")
                    print(
                        f"Output[{i}].shape: {scale_output.shape},\
                        Target[{i}].shape: {scale_target.shape}"
                    )
                    raise e  # Re-raise for debugging

            loss.backward()
            optimizer.step()

            # Update progress bar and accumulate epoch loss
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Display average loss per epoch
        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}],\
            Average Loss: {epoch_loss / len(train_loader):.4f}"
        )

        # Save checkpoint
        if config.SAVE_CHECKPOINT:
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.CHECKPOINTS_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
                ),
            )


if __name__ == "__main__":
    train()
