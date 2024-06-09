# models/yolo_loss.py

import torch
import torch.nn as nn
import numpy as np


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
        # Debugging: Print initial shapes
        print("Initial shapes:")
        print("Predictions shape:", predictions.shape)
        print("Targets shape:", targets.shape)
        print("Anchors shape:", self.anchors.shape)

        # Reshape and permute for easier handling of predictions
        batch_size, grid_size = predictions.size(0), predictions.size(2)
        num_anchors, bbox_attrs = 3, 5 + self.num_classes
        predictions = predictions.view(
            batch_size, num_anchors, bbox_attrs, grid_size, grid_size
        )
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        print("Reshaped Predictions shape:", predictions.shape)

        # Get predicted values
        x = torch.sigmoid(predictions[..., 0])
        y = torch.sigmoid(predictions[..., 1])
        w = predictions[..., 2]
        h = predictions[..., 3]
        conf = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:])

        # Debugging: Shapes
        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)
        print("Shape of w:", w.shape)
        print("Shape of h:", h.shape)
        print("Shape of conf:", conf.shape)
        print("Shape of pred_cls:", pred_cls.shape)

        # Debugging: Print predictions min/max
        print(f"Prediction x min/max: {x.min().item()}/{x.max().item()}")
        print(f"Prediction y min/max: {y.min().item()}/{y.max().item()}")
        print(f"Prediction w min/max: {w.min().item()}/{w.max().item()}")
        print(f"Prediction h min/max: {h.min().item()}/{h.max().item()}")
        print(
            f"Prediction conf min/max: {conf.min().item()}/{conf.max().item()}"
        )
        print(
            f"Prediction cls min/max: {pred_cls.min().item()}/{pred_cls.max().item()}"
        )

        # Scale anchors
        stride = self.img_size / grid_size
        scaled_anchors = self.anchors / stride
        print("Scaled anchors shape:", scaled_anchors.shape)

        # Anchor widths and heights reshaping
        anchor_w = scaled_anchors[:, :, 0:1].view(1, 3, 3, 1, 1)
        anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 3, 1, 1)
        print("Anchor widths shape:", anchor_w.shape)
        print("Anchor heights shape:", anchor_h.shape)

        # Apply width and height scaling
        w = torch.exp(w) * anchor_w
        h = torch.exp(h) * anchor_h

        # Final checks on scaled predictions
        print(
            f"Scaled Prediction w min/max after exp: {w.min().item()}/{w.max().item()}"
        )
        print(
            f"Scaled Prediction h min/max after exp: {h.min().item()}/{h.max().item()}"
        )

        # Masks
        obj_mask = targets[..., 4] == 1
        noobj_mask = targets[..., 4] == 0
        print("Object mask shape:", obj_mask.shape)
        print("No object mask shape:", noobj_mask.shape)

        # Target values
        tx, ty = targets[..., 0], targets[..., 1]
        tw, th = targets[..., 2], targets[..., 3]
        tconf, tcls = targets[..., 4], targets[..., 5:]

        # Debugging: Print targets min/max
        print(f"Target x min/max: {tx.min().item()}/{tx.max().item()}")
        print(f"Target y min/max: {ty.min().item()}/{ty.max().item()}")
        print(f"Target w min/max: {tw.min().item()}/{tw.max().item()}")
        print(f"Target h min/max: {th.min().item()}/{th.max().item()}")
        print(
            f"Target conf min/max: {tconf.min().item()}/{tconf.max().item()}"
        )

        # Debugging: Shapes
        print("Target tx shape:", tx.shape)
        print("Target ty shape:", ty.shape)
        print("Target tw shape:", tw.shape)
        print("Target th shape:", th.shape)
        print("Target conf shape:", tconf.shape)
        print("Target class shape:", tcls.shape)

        # # Reshape the masks to align with multi-scale anchors
        # print("Reshaping obj mask")
        # obj_mask = obj_mask.unsqueeze(2)  # Shape [1, 3, 1, 13, 13]
        # print("Reshaping noobj mask")
        # noobj_mask = noobj_mask.unsqueeze(2)  # Shape [1, 3, 1, 13, 13]

        # Coordinate losses
        print(f"X.shape: {x.shape}")
        print(f"obj_mask shape: {obj_mask.shape}")
        print(f"tx shape: {tx.shape}")
        print(f"self.lambda_coord: {self.lambda_coord}")
        print("Calculating loss_x")
        loss_x = self.lambda_coord * self.mse_loss(x[obj_mask], tx[obj_mask])
        print(f"loss_x: {loss_x}")
        print(f"Y.shape: {y.shape}")
        print(f"obj_mask shape: {obj_mask.shape}")
        print(f"ty shape: {ty.shape}")
        print(f"self.lambda_coord: {self.lambda_coord}")
        print("Calculating loss_y")
        loss_y = self.lambda_coord * self.mse_loss(y[obj_mask], ty[obj_mask])
        print(f"loss_y: {loss_y}")

        # Width and height losses

        print("Calculating loss_w")
        print(f"W.shape: {w.shape}")
        print(f"obj_mask shape: {obj_mask.shape}")
        print(f"tw shape: {tw.shape}")

        loss_w = 0
        loss_h = 0

        for anchor_idx in range(
            w.shape[2]
        ):  # Loop over the anchor dimension (3 in this case)
            print(f"Calculating loss_w for anchor index: {anchor_idx}")
            # Extract the specific anchor slice for w and h
            w_anchor = w[:, :, anchor_idx, :, :][
                obj_mask
            ]  # Shape should match `tw[obj_mask]`
            h_anchor = h[:, :, anchor_idx, :, :][obj_mask]

            # Compute the loss for this specific anchor
            loss_w += self.mse_loss(w_anchor, tw[obj_mask])
            loss_h += self.mse_loss(h_anchor, th[obj_mask])

        print(f"Total loss_w: {loss_w}")
        print(f"Total loss_h: {loss_h}")

        # Object and non-object confidence losses
        print("Calculating loss_conf_obj")
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        print("Calculating loss_conf_noobj")
        loss_conf_noobj = self.lambda_noobj * self.bce_loss(
            conf[noobj_mask], tconf[noobj_mask]
        )

        # Classification loss
        print("Calculating loss_cls")
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

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


if __name__ == "__main__":
    test_yolo_loss_np()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class YOLOLoss(nn.Module):
#     def __init__(self, anchors, num_classes, img_size):
#         super(YOLOLoss, self).__init__()
#         self.anchors = torch.tensor(anchors, dtype=torch.float32)
#         self.num_classes = num_classes
#         self.img_size = img_size
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCELoss()
#         self.lambda_coord = 5
#         self.lambda_noobj = 0.5

#     def forward(self, predictions, targets, anchors):
#         if isinstance(anchors, np.ndarray):
#             anchors = torch.tensor(
#                 anchors, dtype=torch.float32, device=predictions.device
#             )
#         elif isinstance(anchors, list):  # Convert list to tensor
#             anchors = torch.tensor(
#                 anchors, dtype=torch.float32, device=predictions.device
#             )

#         # Debugging: Print initial shapes
#         print("Initial shapes:")
#         print("Predictions shape:", predictions.shape)
#         print("Targets shape:", targets.shape)
#         print("Anchors shape:", anchors.shape)

#         # Convert anchors to [3, 3, 2] if required
#         if anchors.dim() == 2 and anchors.size(1) == 4:
#             anchors = anchors[:, 2:]  # Keep only the width and height
#             anchors = anchors.view(3, 3, 2)
#         elif anchors.dim() == 2 and anchors.size(0) == 3:  # [3, 2] case
#             anchors = anchors  # No further changes needed
#         print("Anchors shape after reshaping:", anchors.shape)

#         # Reshape and permute for easier handling of predictions
#         batch_size, grid_size = predictions.size(0), predictions.size(2)
#         num_anchors, bbox_attrs = len(anchors), 5 + self.num_classes
#         predictions = predictions.view(
#             batch_size, num_anchors, bbox_attrs, grid_size, grid_size
#         )
#         predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
#         print("Reshaped Predictions shape:", predictions.shape)

#         # Get predicted values
#         x = torch.sigmoid(predictions[..., 0])
#         y = torch.sigmoid(predictions[..., 1])
#         w = predictions[..., 2]
#         h = predictions[..., 3]
#         conf = torch.sigmoid(predictions[..., 4])
#         pred_cls = torch.sigmoid(predictions[..., 5:])

#         # Debugging: Shapes
#         print("Shape of x:", x.shape)
#         print("Shape of y:", y.shape)
#         print("Shape of w:", w.shape)
#         print("Shape of h:", h.shape)
#         print("Shape of conf:", conf.shape)
#         print("Shape of pred_cls:", pred_cls.shape)

#         # Debugging: Print predictions min/max
#         print(f"Prediction x min/max: {x.min().item()}/{x.max().item()}")
#         print(f"Prediction y min/max: {y.min().item()}/{y.max().item()}")
#         print(f"Prediction w min/max: {w.min().item()}/{w.max().item()}")
#         print(f"Prediction h min/max: {h.min().item()}/{h.max().item()}")
#         print(
#             f"Prediction conf min/max: {conf.min().item()}/{conf.max().item()}"
#         )
#         print(
#             f"Prediction cls min/max: {pred_cls.min().item()}/{pred_cls.max().item()}"
#         )

#         # Scale anchors
#         print("Stride")
#         stride = self.img_size / grid_size
#         print("scaled_anchors")
#         scaled_anchors = self.anchors / stride

#         # Check the shape of scaled_anchors and reshape to [3, 3, 2] if needed
#         print("scaled_anchors reshaping")
#         print("scaled_anchors: ", scaled_anchors)
#         print("scaled_anchors shape: ", scaled_anchors.shape)
#         if scaled_anchors.shape == (
#             9,
#             4,
#         ):  # This indicates 9 anchors with 4 values each (x, y, width, height)
#             scaled_anchors = scaled_anchors[
#                 :, 2:
#             ]  # Keep only width and height
#             scaled_anchors = scaled_anchors.view(
#                 3, 3, 2
#             )  # Reshape to [3, 3, 2] for multi-scale
#         print("After reshaping::")
#         print("scaled_anchors: ", scaled_anchors)
#         print("scaled_anchors shape: ", scaled_anchors.shape)

#         # # Reshape anchors for YOLO multi-scale compatibility
#         # if scaled_anchors.dim() == 2:  # Case for [3, 2] format
#         #     print("scaled_anchors reshaping")
#         #     print("scaled_anchors: ", scaled_anchors)
#         #     print("scaled_anchors shape: ", scaled_anchors.shape)
#         #     scaled_anchors = scaled_anchors.view(
#         #         1, 3, 2
#         #     )  # Adjust to [1, 3, 2]
#         # elif (
#         #     scaled_anchors.dim() == 3
#         # ):  # Case for [3, 3, 2] format (already structured for scales)
#         #     scaled_anchors = scaled_anchors  # No changes needed

#         # # Define anchor width and height based on the modified shape
#         # if scaled_anchors.shape == (
#         #     1,
#         #     3,
#         #     2,
#         # ):  # Single scale case (without multiple scales)
#         #     print("anchor_w")
#         #     anchor_w = scaled_anchors[:, :, 0:1].view(1, 3, 1, 1)
#         #     print("anchor_h")
#         #     anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 1, 1)
#         # elif scaled_anchors.shape == (3, 3, 2):  # Multi-scale case
#         #     print("anchor_w")
#         #     anchor_w = scaled_anchors[:, :, 0:1].view(1, 3, 3, 1, 1)
#         #     print("anchor_h")
#         #     anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 3, 1, 1)

#         # # Reshape scaled_anchors to (3, 3, 2) if using 3 scales and 3 anchors per scale
#         # if scaled_anchors.shape == (9, 2):  # Check if we have all 9 anchors
#         #     print("scaled_anchors reshaping")
#         #     scaled_anchors = scaled_anchors.view(3, 3, 2)

#         # # print("anchor_w")
#         # # anchor_w = scaled_anchors[:, 0:1].view(1, num_anchors, 1, 1)
#         # # print("anchor_h")
#         # # anchor_h = scaled_anchors[:, 1:2].view(1, num_anchors, 1, 1)

#         # # Extract widths and heights for each anchor scale
#         # print("anchor_w")
#         # anchor_w = scaled_anchors[:, :, 0:1].view(
#         #     1, 3, 3, 1, 1
#         # )  # [1, scales, anchors_per_scale, 1, 1]
#         # print("anchor_h")
#         # anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 3, 1, 1)

#         # Define anchor width and height tensors
#         print("anchor_w")
#         anchor_w = scaled_anchors[:, :, 0:1].view(
#             1, 3, 3, 1, 1
#         )  # Shape [1, 3, 3, 1, 1]
#         print("anchor_h")
#         anchor_h = scaled_anchors[:, :, 1:2].view(1, 3, 3, 1, 1)

#         # Verify shapes
#         print("Scaled anchors shape:", scaled_anchors.shape)
#         print("Anchor widths shape:", anchor_w.shape)
#         print("Anchor heights shape:", anchor_h.shape)

#         # Apply width and height scaling
#         print("w")
#         w = torch.exp(w) * anchor_w
#         print("h")
#         # print(f"h: {h}")
#         print(f"h.shape : {h.shape}")
#         # print(f"anchor_h: {anchor_h}")
#         print(f"anchor_h.shape : {anchor_h.shape}")
#         h = torch.exp(h) * anchor_h

#         # Debugging: After scaling
#         print(
#             f"Scaled Prediction w min/max after exp: {w.min().item()}/{w.max().item()}"
#         )
#         print(
#             f"Scaled Prediction h min/max after exp: {h.min().item()}/{h.max().item()}"
#         )

#         print("Scaled anchors shape:", scaled_anchors.shape)
#         print("Anchor widths shape:", anchor_w.shape)
#         print("Anchor heights shape:", anchor_h.shape)

#         # Masks
#         obj_mask = targets[..., 4] == 1
#         noobj_mask = targets[..., 4] == 0
#         print("Object mask shape:", obj_mask.shape)
#         print("No object mask shape:", noobj_mask.shape)

#         # Target values
#         tx, ty = targets[..., 0], targets[..., 1]
#         tw, th = targets[..., 2], targets[..., 3]
#         tconf, tcls = targets[..., 4], targets[..., 5:]

#         # Debugging: Print targets min/max
#         print(f"Target x min/max: {tx.min().item()}/{tx.max().item()}")
#         print(f"Target y min/max: {ty.min().item()}/{ty.max().item()}")
#         print(f"Target w min/max: {tw.min().item()}/{tw.max().item()}")
#         print(f"Target h min/max: {th.min().item()}/{th.max().item()}")
#         print(
#             f"Target conf min/max: {tconf.min().item()}/{tconf.max().item()}"
#         )

#         # Debugging: Shapes
#         print("Target tx shape:", tx.shape)
#         print("Target ty shape:", ty.shape)
#         print("Target tw shape:", tw.shape)
#         print("Target th shape:", th.shape)
#         print("Target conf shape:", tconf.shape)
#         print("Target class shape:", tcls.shape)

#         # Reshape the masks to align with multi-scale anchors
#         print("Reshaping obj mask")
#         obj_mask = obj_mask.unsqueeze(
#             2
#         )  # Shape [1, 3, 1, 13, 13] -> aligns with [1, 3, 3, 13, 13]
#         print("Reshaping noobj mask")
#         noobj_mask = noobj_mask.unsqueeze(2)  # Shape [1, 3, 1, 13, 13]

#         # Coordinate losses
#         print("Calculating loss_x")
#         loss_x = self.lambda_coord * self.mse_loss(x[obj_mask], tx[obj_mask])
#         print("Calculating loss_y")
#         loss_y = self.lambda_coord * self.mse_loss(y[obj_mask], ty[obj_mask])

#         # Width and height losses
#         print("Calculating loss_w")
#         loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
#         print("Calculating loss_h")
#         loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

#         # Object and non-object confidence losses
#         print("loss_conf_obj")
#         loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
#         if noobj_mask.sum() > 0:
#             loss_conf_noobj = self.lambda_noobj * self.bce_loss(
#                 conf[noobj_mask], tconf[noobj_mask]
#             )
#         else:
#             loss_conf_noobj = torch.tensor(0.0, requires_grad=True)

#         # Classification loss
#         loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

#         # Debugging: Loss components
#         print(f"loss_x: {loss_x.item() if not torch.isnan(loss_x) else 'NaN'}")
#         print(f"loss_y: {loss_y.item() if not torch.isnan(loss_y) else 'NaN'}")
#         print(f"loss_w: {loss_w.item() if not torch.isnan(loss_w) else 'NaN'}")
#         print(f"loss_h: {loss_h.item() if not torch.isnan(loss_h) else 'NaN'}")
#         print(
#             f"loss_conf_obj: {loss_conf_obj.item() if not torch.isnan(loss_conf_obj) else 'NaN'}"
#         )
#         print(
#             f"loss_conf_noobj: {loss_conf_noobj.item() if not torch.isnan(loss_conf_noobj) else 'NaN'}"
#         )
#         print(
#             f"loss_cls: {loss_cls.item() if not torch.isnan(loss_cls) else 'NaN'}"
#         )

#         # Total loss
#         total_loss = (
#             loss_x
#             + loss_y
#             + loss_w
#             + loss_h
#             + loss_conf_obj
#             + loss_conf_noobj
#             + loss_cls
#         )

#         if torch.isnan(total_loss):
#             print("NaN detected in total loss; replacing with 0.")
#             total_loss = torch.tensor(0.0, requires_grad=True)

#         return (
#             total_loss,
#             loss_x,
#             loss_y,
#             loss_w,
#             loss_h,
#             loss_conf_obj,
#             loss_conf_noobj,
#             loss_cls,
#         )


# # Test function to validate YOLOLoss
# def test_yolo_loss():
#     print("Testing func test_yolo_loss")
#     num_classes = 20
#     img_size = 416
#     anchors = [[10, 10], [20, 20], [30, 30]]  # Anchor sizes for testing
#     anchors = torch.tensor(anchors, dtype=torch.float32)
#     # Initialize loss function
#     criterion = YOLOLoss(
#         anchors=anchors, num_classes=num_classes, img_size=img_size
#     )

#     # Mock predictions (batch_size=1, num_anchors=3, grid_size=13, bbox_attrs=25)
#     predictions = torch.zeros((1, 3 * (5 + num_classes), 13, 13)) + 0.01
#     targets = torch.zeros((1, 3, 13, 13, 5 + num_classes))

#     # Assign dummy values to simulate an object
#     # Setting a single object in grid cell (6,6) for anchor 0
#     targets[0, 0, 6, 6, 0] = 0.5  # Target x
#     targets[0, 0, 6, 6, 1] = 0.5  # Target y
#     targets[0, 0, 6, 6, 2] = 0.2  # Target width
#     targets[0, 0, 6, 6, 3] = 0.2  # Target height
#     targets[0, 0, 6, 6, 4] = 1.0  # Object confidence
#     targets[0, 0, 6, 6, 5] = 1.0  # Class 0 probability

#     # Run the forward function
#     try:
#         (
#             loss,
#             loss_x,
#             loss_y,
#             loss_w,
#             loss_h,
#             loss_conf_obj,
#             loss_conf_noobj,
#             loss_cls,
#         ) = criterion(predictions, targets, anchors)
#         print(f"Loss calculated: {loss.item()}")
#         print(
#             "loss_x, loss_y, loss_w, loss_h, loss_conf_obj, loss_conf_noobj, loss_cls"
#         )
#         print(
#             loss_x,
#             loss_y,
#             loss_w,
#             loss_h,
#             loss_conf_obj,
#             loss_conf_noobj,
#             loss_cls,
#         )
#     except Exception as e:
#         print(f"Error in YOLOLoss calculation: {e}")

#     print("-----------------------------------------")


# # test function in yolo_loss.py
# def test_yolo_loss_np():
#     print("Testing func test_yolo_loss_np")
#     # test function for np array anchors in yolo_loss.py
#     num_classes = 20
#     img_size = 416
#     # anchors = np.array([[10, 10], [20, 20], [30, 30]])  # Example anchor sizes
#     anchors = np.array(
#         [
#             [0, 0, 10, 13],
#             [0, 0, 16, 30],
#             [0, 0, 33, 23],
#             [0, 0, 30, 61],
#             [0, 0, 62, 45],
#             [0, 0, 59, 119],
#             [0, 0, 116, 90],
#             [0, 0, 156, 198],
#             [0, 0, 373, 326],
#         ]
#     )

#     # Initialize the loss function
#     criterion = YOLOLoss(
#         anchors=anchors, num_classes=num_classes, img_size=img_size
#     )

#     # Mock predictions (batch_size=1, num_anchors=3, grid_size=13, bbox_attrs=25)
#     predictions = torch.full((1, 3 * (5 + num_classes), 13, 13), 0.01)

#     # Mock targets with objects
#     targets = torch.zeros((1, 3, 13, 13, 5 + num_classes))
#     targets[..., 4] = (
#         1  # Set objectness to 1 to simulate objects in grid cells
#     )
#     targets[..., 0] = 0.5  # Example center x
#     targets[..., 1] = 0.5  # Example center y
#     targets[..., 2] = 0.2  # Example width
#     targets[..., 3] = 0.2  # Example height

#     # Run forward function
#     try:
#         (
#             loss,
#             loss_x,
#             loss_y,
#             loss_w,
#             loss_h,
#             loss_conf_obj,
#             loss_conf_noobj,
#             loss_cls,
#         ) = criterion(predictions, targets, anchors)

#         print(f"Loss calculated: {loss.item()}")
#         print(
#             "loss_x, loss_y, loss_w, loss_h, loss_conf_obj, loss_conf_noobj, loss_cls"
#         )
#         print(
#             loss_x,
#             loss_y,
#             loss_w,
#             loss_h,
#             loss_conf_obj,
#             loss_conf_noobj,
#             loss_cls,
#         )
#     except Exception as e:
#         print(f"Error in YOLOLoss calculation: {e}")
#     print("-----------------------------------------")


# if __name__ == "__main__":
#     test_yolo_loss()
#     test_yolo_loss_np()
