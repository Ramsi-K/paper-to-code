# utils/utils.py

import torch


def intersection_over_union(box1, box2, box_format="corners"):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1 (list or array): Coordinates of the first bounding box.
                              Expected format depends on `box_format`:
                              - "corners": [x1, y1, x2, y2]
                              - "midpoint": [x_center, y_center, width, height]
        box2 (list or array): Coordinates of the second bounding box in the
                              same format as `box1`.
        box_format (str): Format of the bounding boxes.
                          Must be either "corners"
                          or "midpoint".

    Returns:
        float: The Intersection over Union (IoU)
               between the two bounding boxes.

    Note:
        - If box coordinates are in "corners" format,
          IoU is calculated directly.
        - If box coordinates are in "midpoint" format,
          the function first converts
          them to "corners" format.
    """

    if box_format == "midpoint":
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
    elif box_format == "corners":
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    return intersection / union


def non_max_suppression(
    bboxes, iou_threshold=0.5, conf_threshold=0.5, box_format="corners"
):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Parameters:
        bboxes (list): List of boxes, each represented as
        [class_pred, prob_score, x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for NMS.
        conf_threshold (float): Confidence score threshold to keep boxes.
        box_format (str): Format of boxes - "midpoint" or "corners".

    Returns:
        list: Bounding boxes after applying NMS.
    """
    assert isinstance(bboxes, list)

    # Filter out boxes below the confidence threshold
    bboxes = [box for box in bboxes if box[1] > conf_threshold]

    # Sort by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # List to store final boxes after NMS
    bboxes_after_nms = []

    while bboxes:
        # Select the box with the highest confidence score
        chosen_box = bboxes.pop(0)

        # Keep only boxes of different classes or with IoU less than threshold
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format
            )
            < iou_threshold
        ]

        # Add chosen box to the list after NMS
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def iou(box1, box2):
    """
    Compute Intersection over Union between two bounding
    boxes in corner format.
    box1: tensor of shape (N, 4), where each box is [x1, y1, x2, y2]
    box2: tensor of shape (M, 4), anchors in [x1, y1, x2, y2] format
    """
    if box1.shape[-1] != 4 or box2.shape[-1] != 4:
        raise ValueError(
            "Boxes must have 4 elements in [x1, y1, x2, y2] format."
        )

    inter_rect_x1 = torch.max(box1[..., 0:1], box2[..., 0:1].T)
    inter_rect_y1 = torch.max(box1[..., 1:2], box2[..., 1:2].T)
    inter_rect_x2 = torch.min(box1[..., 2:3], box2[..., 2:3].T)
    inter_rect_y2 = torch.min(box1[..., 3:4], box2[..., 3:4].T)

    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1, min=0
    ) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area

    return inter_area / union_area


def parse_anchors(anchor_list):
    """
    Convert anchor list to tensor format.
    """
    anchors = torch.tensor(anchor_list)
    return anchors
