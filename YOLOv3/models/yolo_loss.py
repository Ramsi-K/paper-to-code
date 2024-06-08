# models/yolo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        stride = self.img_size / grid_size
        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes  # 5: tx, ty, tw, th, objectness

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

        # Scale anchors based on stride
        scaled_anchors = self.anchors / stride
        anchor_w = scaled_anchors[:, 0:1].view(1, num_anchors, 1, 1)
        anchor_h = scaled_anchors[:, 1:2].view(1, num_anchors, 1, 1)

        # Adjust predictions based on anchors
        w = torch.exp(w) * anchor_w
        h = torch.exp(h) * anchor_h

        # Ground truth parsing
        obj_mask = targets[..., 4] == 1  # Mask for objects
        noobj_mask = targets[..., 4] == 0  # Mask for no objects

        tx = targets[..., 0]  # Target x
        ty = targets[..., 1]  # Target y
        tw = targets[..., 2]  # Target width
        th = targets[..., 3]  # Target height
        tconf = targets[..., 4]  # Target confidence
        tcls = targets[..., 5:]  # Target class probabilities

        # Loss calculations
        loss_x = self.lambda_coord * self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.lambda_coord * self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.lambda_noobj * self.bce_loss(
            conf[noobj_mask], tconf[noobj_mask]
        )
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        total_loss = (
            loss_x
            + loss_y
            + loss_w
            + loss_h
            + loss_conf_obj
            + loss_conf_noobj
            + loss_cls
        )

        return total_loss
