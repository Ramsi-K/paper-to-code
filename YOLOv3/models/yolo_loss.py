# models/yolo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YOLOLoss(nn.Module):
    """
    Compute the YOLOv3 loss by matching predictions with ground truth targets.
    """

    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.lambda_coord = 5  # Coordinate loss multiplier
        self.lambda_noobj = 0.5  # No-object confidence multiplier

    def forward(self, predictions, targets, anchors):
        """
        Calculate the YOLOv3 loss at a specific scale for predictions and targets.
        """
        if isinstance(anchors, np.ndarray):
            anchors = torch.tensor(
                anchors, dtype=torch.float32, device=predictions.device
            )
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        stride = self.img_size / grid_size
        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes  # 5: tx, ty, tw, th, objectness

        # Debugging
        if torch.isnan(predictions).any():
            print("NaN detected in predictions before loss calculations")
            return torch.tensor(float("nan"))  # Early exit if NaN is found

        # Reshape predictions to (batch_size, num_anchors, grid_size, grid_size, bbox_attrs)
        predictions = predictions.view(
            batch_size, num_anchors, bbox_attrs, grid_size, grid_size
        )
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        # Get predicted components
        x = torch.sigmoid(predictions[..., 0])  # Center x
        y = torch.sigmoid(predictions[..., 1])  # Center y
        w = predictions[..., 2]  # Width (raw)
        h = predictions[..., 3]  # Height (raw)
        conf = torch.sigmoid(predictions[..., 4])  # Objectness score
        pred_cls = torch.sigmoid(predictions[..., 5:])  # Class predictions

        # Debugging
        print(f"Prediction x min/max: {x.min()}/{x.max()}")
        print(f"Prediction y min/max: {y.min()}/{y.max()}")
        print(f"Prediction w min/max: {w.min()}/{w.max()}")
        print(f"Prediction h min/max: {h.min()}/{h.max()}")
        print(f"Prediction conf min/max: {conf.min()}/{conf.max()}")
        print(f"Prediction cls min/max: {pred_cls.min()}/{pred_cls.max()}")

        # Scale anchors based on stride
        scaled_anchors = self.anchors / stride
        anchor_w = scaled_anchors[:, 0:1].view(1, num_anchors, 1, 1)
        anchor_h = scaled_anchors[:, 1:2].view(1, num_anchors, 1, 1)

        # # Adjust predictions based on anchors
        # w = torch.exp(w) * anchor_w
        # h = torch.exp(h) * anchor_h
        # Clamp
        w = torch.clamp(torch.exp(w) * (anchor_w + 1e-8), min=1e-4, max=1e4)
        h = torch.clamp(torch.exp(h) * (anchor_h + 1e-8), min=1e-4, max=1e4)

        # Ground truth parsing
        obj_mask = targets[..., 4] == 1  # Mask for objects
        noobj_mask = targets[..., 4] == 0  # Mask for no objects

        # Adjust loss_conf_noobj calculation
        if noobj_mask.sum() > 0:
            loss_conf_noobj = self.lambda_noobj * self.bce_loss(
                conf[noobj_mask], tconf[noobj_mask]
            )
        else:
            loss_conf_noobj = torch.tensor(
                0.0, requires_grad=True
            )  # Set to 0 if no valid entries

        # Debugging
        print("Predictions shape:", predictions.shape)
        print("Targets shape:", targets.shape)
        print("Scaled anchors shape:", scaled_anchors.shape)

        if obj_mask.sum() == 0:
            print(
                "Warning: Object mask has no targets, skipping object losses."
            )
            loss_x, loss_y, loss_w, loss_h, loss_conf_obj, loss_cls = (
                0,
                0,
                0,
                0,
                0,
                0,
            )

        tx = targets[..., 0]  # Target x
        ty = targets[..., 1]  # Target y
        tw = targets[..., 2]  # Target width
        th = targets[..., 3]  # Target height
        tconf = targets[..., 4]  # Target confidence
        tcls = targets[..., 5:]  # Target class probabilities

        # Debugging
        print(f"Target x: {tx[obj_mask]}")
        print(f"Target y: {ty[obj_mask]}")
        print(f"Target w: {tw[obj_mask]}")
        print(f"Target h: {th[obj_mask]}")
        print(f"Object mask sum: {obj_mask.sum()}")

        print(f"Target tx (any NaN?): {torch.isnan(tx).any()}")
        print(f"Target tw (any NaN?): {torch.isnan(tw).any()}")
        print(f"Target tconf (any NaN?): {torch.isnan(tconf).any()}")

        # Loss calculations
        # loss_x = self.lambda_coord * self.mse_loss(x[obj_mask], tx[obj_mask])
        # loss_y = self.lambda_coord * self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_x = self.lambda_coord * self.mse_loss(
            x[obj_mask], tx[obj_mask] + 1e-8
        )
        loss_y = self.lambda_coord * self.mse_loss(
            y[obj_mask], ty[obj_mask] + 1e-8
        )

        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.lambda_noobj * self.bce_loss(
            conf[noobj_mask], tconf[noobj_mask]
        )
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        # Debugging
        print(
            f"loss_x: {loss_x.item() if loss_x.item() == loss_x.item() else 'NaN'}"
        )
        print(
            f"loss_y: {loss_y.item() if loss_y.item() == loss_y.item() else 'NaN'}"
        )
        print(
            f"loss_w: {loss_w.item() if loss_w.item() == loss_w.item() else 'NaN'}"
        )
        print(
            f"loss_h: {loss_h.item() if loss_h.item() == loss_h.item() else 'NaN'}"
        )
        print(
            f"loss_conf_obj: {loss_conf_obj.item() if loss_conf_obj.item() == loss_conf_obj.item() else 'NaN'}"
        )
        print(
            f"loss_conf_noobj: {loss_conf_noobj.item() if loss_conf_noobj.item() == loss_conf_noobj.item() else 'NaN'}"
        )
        print(
            f"loss_cls: {loss_cls.item() if loss_cls.item() == loss_cls.item() else 'NaN'}"
        )

        total_loss = (
            loss_x
            + loss_y
            + loss_w
            + loss_h
            + loss_conf_obj
            + loss_conf_noobj
            + loss_cls
        )

        if torch.isnan(total_loss):
            print("NaN detected in total loss; replacing with 0.")
            total_loss = torch.tensor(0.0, requires_grad=True)

        return (
            total_loss,
            loss_x,
            loss_y,
            loss_w,
            loss_h,
            loss_conf_obj,
            loss_conf_noobj,
            loss_cls,
        )


# Test function to validate YOLOLoss
def test_yolo_loss():
    num_classes = 20
    img_size = 416
    anchors = [[10, 10], [20, 20], [30, 30]]  # Anchor sizes for testing

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


# test function in yolo_loss.py
def test_yolo_loss_np():
    # test function for np array anchors in yolo_loss.py
    num_classes = 20
    img_size = 416
    anchors = np.array([[10, 10], [20, 20], [30, 30]])  # Example anchor sizes

    # Initialize the loss function
    criterion = YOLOLoss(
        anchors=anchors, num_classes=num_classes, img_size=img_size
    )

    # Mock predictions (batch_size=1, num_anchors=3, grid_size=13, bbox_attrs=25)
    predictions = torch.full((1, 3 * (5 + num_classes), 13, 13), 0.01)

    # Mock targets with objects
    targets = torch.zeros((1, 3, 13, 13, 5 + num_classes))
    targets[..., 4] = (
        1  # Set objectness to 1 to simulate objects in grid cells
    )
    targets[..., 0] = 0.5  # Example center x
    targets[..., 1] = 0.5  # Example center y
    targets[..., 2] = 0.2  # Example width
    targets[..., 3] = 0.2  # Example height

    # Run forward function
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


if __name__ == "__main__":
    test_yolo_loss()
    # test_yolo_loss_np()
