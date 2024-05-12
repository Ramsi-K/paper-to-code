import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    IMG_DIR,
    LABEL_DIR,
    test_transforms,
    ANCHORS,
    CONF_THRESHOLD,
    NMS_IOU_THRESH,
    MAP_IOU_THRESH,
)
from utils import non_max_suppression, mean_average_precision, cells_to_bboxes
from dataset import YOLODataset
from torch.utils.data import DataLoader


class CNNBlock(nn.Module):
    """
    A CNN block consisting of convolution, batch normalization, and Leaky ReLU activation.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_act (bool): If True, applies batch normalization and activation; otherwise, only convolution.
    """

    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, bias=not bn_act, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn_act else nn.Identity()
        self.leaky = nn.LeakyReLU(0.1) if bn_act else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection.

    Parameters:
        channels (int): Number of input and output channels.
        use_residual (bool): If True, adds a skip connection; otherwise, behaves as a sequential block.
        num_repeats (int): Number of times to repeat this block.
    """

    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.num_repeats = num_repeats
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(
                        channels // 2, channels, kernel_size=3, padding=1
                    ),
                )
            ]
        self.use_residual = use_residual
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    """
    Prediction head for a specific scale, responsible for bounding box predictions.

    Parameters:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels,
                (num_classes + 5) * 3,
                bn_act=False,
                kernel_size=1,
            ),
        )
        self.num_classes = num_classes
        self.out_channels = (num_classes + 5) * 3

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(
                x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]
            )
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers, self.layer_output_channels = self._create_conv_layers()

    def _reduce_channels(self, x, target_channels):
        """Reduce the number of channels to match target_channels."""
        if x.shape[1] != target_channels:
            conv_layer = nn.Conv2d(
                x.shape[1],
                target_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ).to(x.device)
            x = nn.LeakyReLU(0.1)(conv_layer(x))
            # print(f"Channel-reduced shape to {target_channels}: {x.shape}")
        return x

    def forward(self, x):
        outputs = []
        route_connections = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                x = self._reduce_channels(
                    x, layer.in_channels
                )  # Ensure the right channel count for ScalePrediction
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                # Ensure spatial dimensions match before concatenation
                if x.shape[2:] != route_connections[-1].shape[2:]:
                    x = F.interpolate(
                        x, size=route_connections[-1].shape[2:], mode="nearest"
                    )
                x = torch.cat([x, route_connections.pop()], dim=1)
                # Print shape after concatenation for debugging
                # print(f"Shape after concatenation at layer {i}: {x.shape}")

                # Reduce channels after concatenation to match expected layer output
                target_channels = self.layer_output_channels[i]
                x = self._reduce_channels(x, target_channels)

                # Print shape after channel reduction for further debugging
                # print(f"Shape after channel reduction at layer {i}: {x.shape}")

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        layer_output_channels = []
        in_channels = self.in_channels

        architecture = [
            (32, 3, 1),
            ResidualBlock(channels=32, num_repeats=1),
            (64, 3, 2),
            ResidualBlock(channels=64, num_repeats=2),
            (128, 3, 2),
            ResidualBlock(channels=128, num_repeats=8),
            (256, 3, 2),
            ResidualBlock(channels=256, num_repeats=8),
            (512, 3, 2),
            ResidualBlock(channels=512, num_repeats=4),
            (1024, 3, 2),
            ResidualBlock(channels=1024, num_repeats=4),
            ScalePrediction(in_channels=1024, num_classes=self.num_classes),
            nn.Upsample(scale_factor=2),
            ScalePrediction(in_channels=512, num_classes=self.num_classes),
            nn.Upsample(scale_factor=2),
            ScalePrediction(in_channels=256, num_classes=self.num_classes),
        ]

        for x in architecture:
            if isinstance(x, tuple):
                layers.append(
                    CNNBlock(
                        in_channels,
                        x[0],
                        kernel_size=x[1],
                        stride=x[2],
                        padding=1,
                    )
                )
                layer_output_channels.append(x[0])  # Store out_channels
                in_channels = x[0]
            else:
                layers.append(x)
                layer_output_channels.append(
                    x.out_channels
                    if hasattr(x, "out_channels")
                    else in_channels
                )

        return layers, layer_output_channels


def test(model, loader, iou_threshold, anchors, threshold, device="cuda"):
    """
    Tests the model on the provided dataset loader and calculates mAP.

    Parameters:
        model (nn.Module): Trained YOLOv3 model.
        loader (DataLoader): DataLoader for the test dataset.
        iou_threshold (float): IoU threshold for mean Average Precision (mAP).
        anchors (list): Anchor boxes.
        threshold (float): Confidence threshold for predictions.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        float: mAP score.
    """
    model.eval()  # Set model to evaluation mode
    pred_boxes, true_boxes = [], []

    for batch_idx, (x, labels) in enumerate(loader):
        if batch_idx > 1:  # Limit to the first two batches
            break

        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):  # For each scale
            S = predictions[i].shape[2]  # Should align with 13, 26, or 52
            # print(f"Processing boxes for scale {i} with grid size {S}")
            anchor = torch.tensor(anchors[i]).to(device) * S

            # Obtain bounding boxes at this scale
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            # print(f"Finished processing boxes for scale {i}")

            # # Check if the format of boxes_scale_i is as expected
            # if isinstance(boxes_scale_i, list) and isinstance(
            #     boxes_scale_i[0], list
            # ):
            #     print(
            #         f"Boxes scale {i} is a nested list structure with {len(boxes_scale_i)} elements."
            #     )
            #     print(
            #         f"Each element contains {len(boxes_scale_i[0])} bounding boxes per batch entry."
            #     )

            #     # Optional: Uncomment the next line to check each bounding box format
            #     print(f"Bounding box example: {boxes_scale_i[0][0]}")

            # Accumulate bounding boxes for each image in the batch
            for idx, box_list in enumerate(boxes_scale_i):
                bboxes[idx].extend(box_list)

        # Process true bounding boxes
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx], iou_threshold=iou_threshold, threshold=threshold
            )
            pred_boxes.extend([[batch_idx] + box for box in nms_boxes])

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    true_boxes.append([batch_idx] + box)

    model.train()  # Set model back to training mode
    filtered_boxes = non_max_suppression(
        pred_boxes, iou_threshold=NMS_IOU_THRESH, threshold=CONF_THRESHOLD
    )
    mAP = mean_average_precision(
        filtered_boxes, true_boxes, iou_threshold=MAP_IOU_THRESH
    )

    return mAP


if __name__ == "__main__":
    # Sample main block to test the model script independently
    model = YOLOv3(num_classes=20).to("cuda")
    test_dataset = YOLODataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
        transform=test_transforms,
        S=[13, 26, 52],
        C=20,
    )
    image, targets = test_dataset[0]
    # print("Image shape:", image.shape)
    # print("Targets:", targets)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    for images, targets in test_loader:
        # print("Batch loaded successfully")
        # print("Image batch shape:", images.shape)
        # print("Targets batch:", targets)
        break

    model = YOLOv3(num_classes=20).to("cuda")
    images, targets = next(iter(test_loader))

    # Check shapes and contents if needed
    # print("Image batch shape:", images.shape)
    # print("Targets batch:", targets)

    images = images.to("cuda")
    with torch.no_grad():
        preds = model(images)

    # print("Model output:", preds)
    # print("Starting test function...")

    mAP = test(
        model=model,
        loader=test_loader,
        iou_threshold=0.5,
        anchors=ANCHORS,
        threshold=0.4,
        device="cuda",
    )
    # print(f"Test mAP: {mAP:.4f}")
    # print("Test function completed.")
