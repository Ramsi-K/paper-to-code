# models/yolo_loss.py

from dataset import VOCDataset
import torch
import torch.nn as nn
import numpy as np
from config import IMAGE_SIZE, ANCHORS, VOC_CLASSES


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        # Hard-code anchors as a torch tensor with the known format
        self.anchors = torch.tensor(anchors, dtype=torch.float32)[:, 2:].view(
            3, 3, 2
        )
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        print(f"Predictions shape before reshape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")

        # Reshape and permute for easier handling of predictions
        batch_size, grid_size = predictions.size(0), predictions.size(2)
        num_anchors, bbox_attrs = 3, 5 + self.num_classes
        predictions = predictions.view(
            batch_size, num_anchors, bbox_attrs, grid_size, grid_size
        )
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        print(f"Predictions shape after permute: {predictions.shape}")

        # Get predicted values
        x = torch.sigmoid(predictions[..., 0])
        y = torch.sigmoid(predictions[..., 1])
        w = predictions[..., 2]
        h = predictions[..., 3]
        conf = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:])

        print(f"x, y shape: {x.shape}, {y.shape}")
        print(f"w, h shape: {w.shape}, {h.shape}")
        print(f"conf shape: {conf.shape}")
        print(f"pred_cls shape: {pred_cls.shape}")

        # Scale anchors
        stride = self.img_size / grid_size
        scaled_anchors = self.anchors / stride

        # Anchor widths and heights reshaping
        anchor_w = scaled_anchors[:, :, 0:1].view(1, 3, 3, 1, 1)
        anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 3, 1, 1)

        print(
            f"Anchor_w shape: {anchor_w.shape},\
            Anchor_h shape: {anchor_h.shape}"
        )

        # Adjust width and height scaling to avoid NaN
        w = torch.exp(w) * anchor_w
        h = torch.exp(h) * anchor_h

        print(f"w after scaling shape: {w.shape}")
        print(f"h after scaling shape: {h.shape}")

        # Masks
        obj_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0

        print(
            f"obj_mask shape: {obj_mask.shape}, noobj_mask shape: {noobj_mask.shape}"
        )

        # Target values
        tx, ty = targets[..., 0], targets[..., 1]
        tw, th = targets[..., 2], targets[..., 3]
        tconf, tcls = targets[..., 4], targets[..., 5:]

        print(f"tx, ty shape: {tx.shape}, {ty.shape}")
        print(f"tw, th shape: {tw.shape}, {th.shape}")
        print(f"tconf shape: {tconf.shape}")
        print(f"tcls shape: {tcls.shape}")

        # Coordinate losses
        loss_x = self.lambda_coord * self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.lambda_coord * self.mse_loss(y[obj_mask], ty[obj_mask])

        # Width and height losses
        loss_w = 0
        loss_h = 0

        for anchor_idx in range(
            w.shape[2]
        ):  # Loop over the anchor dimension (3 in this case)
            # Extract the specific anchor slice for w and h
            w_anchor = w[:, :, anchor_idx, :, :][
                obj_mask
            ]  # Shape should match `tw[obj_mask]`
            h_anchor = h[:, :, anchor_idx, :, :][obj_mask]

            print(f"w_anchor shape: {w_anchor.shape}")
            print(f"h_anchor shape: {h_anchor.shape}")
            print(f"tw[obj_mask] shape: {tw[obj_mask].shape}")
            print(f"th[obj_mask] shape: {th[obj_mask].shape}")

            # Compute the loss for this specific anchor
            loss_w += self.mse_loss(w_anchor, tw[obj_mask])
            loss_h += self.mse_loss(h_anchor, th[obj_mask])

        # Object and no-object confidence losses
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.lambda_noobj * self.bce_loss(
            conf[noobj_mask], tconf[noobj_mask]
        )
        print(
            f"loss_conf_obj: {loss_conf_obj}, loss_conf_noobj: {loss_conf_noobj}"
        )

        # Classification loss
        tcls = tcls.expand_as(pred_cls)
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        print(f"loss_cls: {loss_cls}")

        # Total loss
        total_loss = (
            loss_x
            + loss_y
            + loss_w
            + loss_h
            + loss_conf_obj
            + loss_conf_noobj
            + loss_cls
        )
        print(f"Total loss: {total_loss.item()}")

        # return (
        #     total_loss,
        #     loss_x,
        #     loss_y,
        #     loss_w,
        #     loss_h,
        #     loss_conf_obj,
        #     loss_conf_noobj,
        #     loss_cls,
        # )

        return total_loss


# Test function to validate YOLOLoss
def test_yolo_loss():
    print("Testing func test_yolo_loss")
    num_classes = 20
    img_size = 416
    anchors = [[10, 10], [20, 20], [30, 30]]  # Anchor sizes for testing
    anchors = torch.tensor(anchors, dtype=torch.float32)
    # Initialize loss function
    criterion = YOLOLoss(
        anchors=anchors, num_classes=num_classes, img_size=img_size
    )

    # Mock predictions (batch_size=1, num_anchors=3, grid_size=13, bbox_attrs=25)
    predictions = torch.zeros((1, 3 * (5 + num_classes), 13, 13)) + 0.01
    targets = torch.zeros((1, 3, 13, 13, 5 + num_classes))

    # Assign dummy values to simulate an object
    # Setting a single object in grid cell (6,6) for anchor 0
    targets[0, 0, 6, 6, 0] = 0.5  # Target x
    targets[0, 0, 6, 6, 1] = 0.5  # Target y
    targets[0, 0, 6, 6, 2] = 0.2  # Target width
    targets[0, 0, 6, 6, 3] = 0.2  # Target height
    targets[0, 0, 6, 6, 4] = 1.0  # Object confidence
    targets[0, 0, 6, 6, 5] = 1.0  # Class 0 probability

    # Run the forward function
    try:
        (
            loss,
            loss_x,
            loss_y,
            loss_w,
            loss_h,
            loss_conf_obj,
            loss_conf_noobj,
            loss_cls,
        ) = criterion(predictions, targets, anchors)
        print(f"Loss calculated: {loss.item()}")
        print(
            "loss_x, loss_y, loss_w, loss_h, loss_conf_obj, loss_conf_noobj, loss_cls"
        )
        print(
            loss_x,
            loss_y,
            loss_w,
            loss_h,
            loss_conf_obj,
            loss_conf_noobj,
            loss_cls,
        )
    except Exception as e:
        print(f"Error in YOLOLoss calculation: {e}")

    print("-----------------------------------------")


# Test function for YOLOLoss with np anchors
def test_yolo_loss_np():
    print("Testing func test_yolo_loss_np")
    num_classes = 20
    img_size = 416
    anchors = np.array(
        [
            [0, 0, 10, 13],
            [0, 0, 16, 30],
            [0, 0, 33, 23],
            [0, 0, 30, 61],
            [0, 0, 62, 45],
            [0, 0, 59, 119],
            [0, 0, 116, 90],
            [0, 0, 156, 198],
            [0, 0, 373, 326],
        ]
    )

    criterion = YOLOLoss(
        anchors=anchors, num_classes=num_classes, img_size=img_size
    )

    predictions = torch.full((1, 3 * (5 + num_classes), 13, 13), 0.01)
    targets = torch.zeros((1, 3, 13, 13, 5 + num_classes))
    targets[..., 4] = 1
    targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3] = (
        0.5,
        0.5,
        0.2,
        0.2,
    )

    try:
        (
            loss,
            loss_x,
            loss_y,
            loss_w,
            loss_h,
            loss_conf_obj,
            loss_conf_noobj,
            loss_cls,
        ) = criterion(predictions, targets)
        print(f"Loss calculated: {loss.item()}")
    except Exception as e:
        print(f"Error in YOLOLoss calculation: {e}")

    print("-----------------------------------------")


def test_yolo_loss_with_voc_dataset():
    print("Testing YOLOLoss with Dummy Targets...")

    # Initialize YOLOLoss with anchor boxes and number of classes
    num_classes = 20  # Define based on your setup
    anchors = [
        [0, 0, 10, 13],
        [0, 0, 16, 30],
        [0, 0, 33, 23],
        [0, 0, 30, 61],
        [0, 0, 62, 45],
        [0, 0, 59, 119],
        [0, 0, 116, 90],
        [0, 0, 156, 198],
        [0, 0, 373, 326],
    ]
    img_size = 416
    yolo_loss = YOLOLoss(
        anchors=anchors, num_classes=num_classes, img_size=img_size
    )

    # Create dummy predictions and targets for testing
    grid_size = 52  # for scale 3
    predictions = torch.full(
        (1, 3 * (5 + num_classes), grid_size, grid_size), 0.01
    )
    targets = torch.zeros((1, 3, grid_size, grid_size, 5 + num_classes))

    # Set dummy values for an object at a specific location
    targets[0, 0, 25, 25, 4] = 1  # Confidence for object presence
    targets[0, 0, 25, 25, 0:4] = torch.tensor(
        [0.5, 0.5, 0.2, 0.2]
    )  # x, y, w, h
    targets[0, 0, 25, 25, 5] = 1  # Set class

    try:
        (
            total_loss,
            loss_x,
            loss_y,
            loss_w,
            loss_h,
            loss_conf_obj,
            loss_conf_noobj,
            loss_cls,
        ) = yolo_loss(predictions, targets)

        # Print the loss breakdown
        print("Loss calculated:", total_loss.item())
        print("Breakdown:")
        print("loss_x:", loss_x.item())
        print("loss_y:", loss_y.item())
        print("loss_w:", loss_w.item())
        print("loss_h:", loss_h.item())
        print("loss_conf_obj:", loss_conf_obj.item())
        print("loss_conf_noobj:", loss_conf_noobj.item())
        print("loss_cls:", loss_cls.item())

    except Exception as e:
        print("Error during YOLOLoss calculation:", e)


if __name__ == "__main__":
    # test_yolo_loss()
    # test_yolo_loss_np()
    test_yolo_loss_with_voc_dataset()
