import torch
import torch.nn as nn
from config import DEVICE
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1  # Object presence mask
        noobj = target[..., 0] == 0  # No object mask

        # Debugging shapes before loss calculations
        print(f"Predictions shape: {predictions.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Object mask shape: {obj.shape}")
        print(f"No-object mask shape: {noobj.shape}")
        print(f"Anchors shape: {anchors.shape}")

        # No-object loss
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj]
        )
        print(f"No-object loss computed: {no_object_loss.item()}")

        # Object loss
        box_preds = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),  # Center coordinates
                torch.exp(predictions[..., 3:5]) * anchors,  # Width and height
            ],
            dim=-1,
        )
        ious = intersection_over_union(
            box_preds[obj], target[..., 1:5][obj]
        ).detach()
        object_loss = self.bce(
            predictions[..., 0:1][obj], ious * target[..., 0:1][obj]
        )
        print(f"Object loss computed: {object_loss.item()}")

        # Box Coordinate loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        print(f"Box loss computed: {box_loss.item()}")

        # Class loss
        if obj.sum() > 0:  # Check to ensure non-zero elements in obj mask
            class_loss = self.entropy(
                predictions[..., 5:][obj], target[..., 5][obj].long()
            )
            print(f"Class loss computed: {class_loss.item()}")
        else:
            class_loss = torch.tensor(0.0, device=DEVICE)
            print("Class loss computed: 0.0 (no objects present)")

        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
        print(f"Total loss computed: {total_loss.item()}")

        return total_loss


# class YOLOLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.entropy = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()

#         # Constants
#         self.lambda_class = 1
#         self.lambda_noobj = 10
#         self.lambda_obj = 1
#         self.lambda_box = 10

#     def forward(self, predictions, target, anchors):
#         obj = target[..., 0] == 1  # Object presence mask
#         noobj = target[..., 0] == 0  # No object mask

#         ## Debugging shapes before loss calculations
#         # print(f"Predictions shape: {predictions.shape}")
#         # print(f"Target shape: {target.shape}")
#         # print(f"Object mask shape: {obj.shape}")
#         # print(f"No-object mask shape: {noobj.shape}")
#         # print(
#         #     f"Anchors shape: {anchors.shape}"
#         # )  # Keep this to verify the shape

#         # No-object loss
#         no_object_loss = self.bce(
#             predictions[..., 0:1][noobj], target[..., 0:1][noobj]
#         )
#         # print(f"No-object loss computed: {no_object_loss.item()}")

#         # Object loss
#         box_preds = torch.cat(
#             [
#                 self.sigmoid(predictions[..., 1:3]),  # Center coordinates
#                 torch.exp(predictions[..., 3:5]) * anchors,  # Width and height
#             ],
#             dim=-1,
#         )
#         ious = intersection_over_union(
#             box_preds[obj], target[..., 1:5][obj]
#         ).detach()
#         object_loss = self.bce(
#             predictions[..., 0:1][obj], ious * target[..., 0:1][obj]
#         )
#         # print(f"Object loss computed: {object_loss.item()}")

#         # Box Coordinate loss
#         predictions[..., 1:3] = self.sigmoid(
#             predictions[..., 1:3]
#         )  # Sigmoid for xy coords
#         target[..., 3:5] = torch.log(
#             1e-6 + target[..., 3:5] / anchors
#         )  # Normalize target box size
#         box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
#         # print(f"Box loss computed: {box_loss.item()}")

#         # Class loss
#         class_loss = self.entropy(
#             predictions[..., 5:][obj], target[..., 5][obj].long()
#         )
#         # print(f"Class loss computed: {class_loss.item()}")

#         total_loss = (
#             self.lambda_box * box_loss
#             + self.lambda_obj * object_loss
#             + self.lambda_noobj * no_object_loss
#             + self.lambda_class * class_loss
#         )
#         # print(f"Total loss computed: {total_loss.item()}")

#         return total_loss


# Define the test function
def test_yolo_loss():
    # Initialize the YOLOLoss object
    loss_fn = YOLOLoss()

    # Set up dummy predictions and targets
    batch_size = 2
    num_anchors = 3
    grid_size = 8
    num_classes = 20

    # Random predictions with a shape similar to what YOLOv3 produces
    predictions = torch.randn(
        (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes),
        device=DEVICE,
    )

    # Random targets with expected YOLOv3 target structure
    targets = torch.zeros(
        (batch_size, num_anchors, grid_size, grid_size, 6), device=DEVICE
    )
    targets[..., 0] = torch.randint(
        0, 2, (batch_size, num_anchors, grid_size, grid_size)
    )  # Objectness
    targets[..., 1:3] = torch.rand(
        (batch_size, num_anchors, grid_size, grid_size, 2)
    )  # Center x, y
    targets[..., 3:5] = (
        torch.rand((batch_size, num_anchors, grid_size, grid_size, 2)) * 2
    )  # Width, height
    targets[..., 5] = torch.randint(
        0, num_classes, (batch_size, num_anchors, grid_size, grid_size)
    )  # Class

    # Example anchors for testing
    anchors = torch.tensor(
        [(10, 13), (16, 30), (33, 23)], dtype=torch.float32, device=DEVICE
    )
    anchors = anchors.reshape(1, num_anchors, 1, 1, 2).to(
        DEVICE
    )  # Reshape as expected in YOLOLoss

    print(f"Predictions shape: {predictions.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Anchors shape after reshape: {anchors.shape}")

    # Calculate the loss
    loss_value = loss_fn(predictions, targets, anchors)
    print(f"Loss value: {loss_value.item()}")


def test_grid_sizes():
    # Initialize the YOLOLoss object
    loss_fn = YOLOLoss()

    # Set up dummy predictions and targets with various grid sizes
    batch_size = 2
    num_anchors = 3
    num_classes = 20
    grid_sizes = [8, 16, 32]  # Different grid sizes

    # Iterate over grid sizes to test each scale independently
    for grid_size in grid_sizes:
        print(f"\nTesting with grid size: {grid_size}")

        # Predictions and targets for this specific grid size
        predictions = torch.randn(
            (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes),
            device=DEVICE,
        )
        targets = torch.zeros(
            (batch_size, num_anchors, grid_size, grid_size, 6), device=DEVICE
        )
        targets[..., 0] = torch.randint(
            0, 2, (batch_size, num_anchors, grid_size, grid_size)
        )  # Objectness
        targets[..., 1:3] = torch.rand(
            (batch_size, num_anchors, grid_size, grid_size, 2)
        )  # Center x, y
        targets[..., 3:5] = (
            torch.rand((batch_size, num_anchors, grid_size, grid_size, 2)) * 2
        )  # Width, height
        targets[..., 5] = torch.randint(
            0, num_classes, (batch_size, num_anchors, grid_size, grid_size)
        )  # Class

        # Example anchors for testing
        anchors = torch.tensor(
            [(10, 13), (16, 30), (33, 23)], dtype=torch.float32, device=DEVICE
        ).reshape(1, num_anchors, 1, 1, 2)

        # Calculate the loss for this grid size
        loss_value = loss_fn(predictions, targets, anchors)
        print(f"Loss for grid size {grid_size}: {loss_value.item()}")


# Run the test function
if __name__ == "__main__":
    test_yolo_loss()
    test_grid_sizes()
