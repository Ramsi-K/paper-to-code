import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def plot_boxes(image_path, boxes, save_path=None):
    """
    Plots bounding boxes on the image and saves it.

    Parameters:
        image_path (str): Path to the original image.
        boxes (list): List of bounding boxes [class_pred, confidence, x1, y1, x2, y2].
        save_path (str, optional): Path to save the annotated image.

    Returns:
        None
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        class_pred, confidence, x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{int(class_pred)}: {confidence:.2f}", fill="red")

    if save_path:
        image.save(save_path)
    else:
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def iou_width_height(boxes1, boxes2):
    """
    Calculates the Intersection over Union (IoU) based on width and height.

    Parameters:
        boxes1 (torch.Tensor): Width and height of the first bounding boxes.
        boxes2 (torch.Tensor): Width and height of the second bounding boxes.

    Returns:
        torch.Tensor: IoU values between the provided boxes.
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1]
        + boxes2[..., 0] * boxes2[..., 1]
        - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates the IoU between predicted and target bounding boxes.

    Parameters:
        boxes_preds (torch.Tensor): Predicted bounding boxes (BATCH_SIZE, 4).
        boxes_labels (torch.Tensor): Target bounding boxes (BATCH_SIZE, 4).
        box_format (str): Format of bounding boxes - "midpoint" or "corners".

    Returns:
        torch.Tensor: IoU values for each pair of predicted and target boxes.
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds[..., :4]
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels[..., :4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(
    bboxes,
    iou_threshold=0.5,
    threshold=0.9,
    box_format="corners",
):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Parameters:
        bboxes (list): List of boxes with [class_pred, prob_score, x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for NMS.
        threshold (float): Confidence score threshold to keep boxes.
        box_format (str): Format of boxes - "midpoint" or "corners".

    Returns:
        list: Bounding boxes after applying NMS.
    """
    assert isinstance(bboxes, list)
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
):
    """
    Calculates the mean Average Precision (mAP) across all classes.

    Parameters:
        pred_boxes (list): Predicted boxes.
        true_boxes (list): Ground truth boxes.
        iou_threshold (float): IoU threshold for a correct prediction.
        box_format (str): Format of boxes - "midpoint" or "corners".
        num_classes (int): Number of classes.

    Returns:
        float: mAP value across all classes.
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [
            detection for detection in pred_boxes if detection[1] == c
        ]
        ground_truths = [
            true_box for true_box in true_boxes if true_box[1] == c
        ]

        # Counts number of ground truth boxes for each image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                gt for gt in ground_truths if gt[0] == detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Converts model predictions or ground truth bounding boxes from grid cell format to full image coordinates.

    Parameters:
        predictions (torch.Tensor): Tensor of size (N, 3, S, S, num_classes + 5) containing bounding boxes.
        anchors (torch.Tensor): Tensor containing anchor boxes.
        S (int): Grid size (height and width) of the feature map.
        is_preds (bool): If True, indicates that `predictions` is model output; else, it's ground truth boxes.

    Returns:
        list: List of converted bounding boxes for each image in the batch.
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    # Debugging output to verify `S` and shape alignment
    print(f"Initial S: {S}, Predictions shape: {predictions.shape}")

    # Dynamically adjust S if it does not match predictions spatial dimensions
    if predictions.shape[2] != S:
        S = predictions.shape[2]
        print(f"Adjusted S to match predictions: {S}")

    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = (
            torch.exp(box_predictions[..., 2:]) * anchors
        )
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Recalculate cell_indices based on updated S
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], num_anchors, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    print(f"cell_indices shape after adjustment: {cell_indices.shape}")

    # Coordinate calculations with adjusted S
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = (
        1
        / S
        * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    )
    w_h = 1 / S * box_predictions[..., 2:4]

    # Concatenate results and reshape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, w_h), dim=-1
    ).reshape(BATCH_SIZE, num_anchors * S * S, 6)

    return converted_bboxes.tolist()


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility.

    Parameters:
        seed (int): Seed value for random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
