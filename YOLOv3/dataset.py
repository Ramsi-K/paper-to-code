import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from config import VOC_CLASSES, ANCHORS, IMAGE_SIZE, train_transforms
from utils.utils import intersection_over_union


def download_voc_dataset(year="2012", root="data/VOC"):
    if os.path.exists(os.path.join(root, "VOCdevkit", f"VOC{year}")):
        print(
            f"Pascal VOC {year} dataset already exists in {root}. Skipping download."
        )
    else:
        VOCDetection(
            root=root,
            year=year,
            image_set="train",
            download=True,
            transform=train_transforms,
        )
        print(f"Pascal VOC {year} dataset downloaded successfully.")


class VOCDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        img_size=IMAGE_SIZE,
        anchors=ANCHORS,
        transform=None,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.S = [13, 26, 52]
        self.anchors = np.array(
            [  # Convert anchors to numpy array for easier handling
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
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.images = sorted(
            [img for img in os.listdir(img_dir) if img.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = image.size
        image = np.array(image)

        label_filename = img_filename.replace(".jpg", ".xml")
        label_path = os.path.join(self.label_dir, label_filename)
        bboxes = self._parse_annotations(label_path, image_width, image_height)

        if self.transform:
            augmentations = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=[box[4] for box in bboxes],
            )
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [
            torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S
        ]

        for box in bboxes:
            # print("Printing box--------------")
            # print(box)
            x_center, y_center, width, height, class_label = box

            # Convert center format to corners for IoU calculations
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            box_corners = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Calculate IoU for each anchor using intersection_over_union
            iou_anchors = np.array(
                [
                    intersection_over_union(box_corners, anchor, "corners")
                    for anchor in self.anchors
                ]
            )

            # Convert the sorted indices to determine the best-matching anchors
            anchor_indices = iou_anchors.argsort()[::-1]
            has_anchor = [False, False, False]

            # print(f"Box: {box}")
            # print(f"Box corners: {box_corners}")
            # print(f"IoU with anchors: {iou_anchors}")
            # print(f"Anchor indices (sorted): {anchor_indices}")

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y_center), int(S * x_center)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x_center - j, S * y_center - i
                    width_cell, height_cell = width * S, height * S
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = (
                        torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(
                        class_label
                    )
                    has_anchor[scale_idx] = True
                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

    def _parse_annotations(self, label_path, image_width, image_height):
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            class_idx = VOC_CLASSES.index(label)
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            boxes.append([x_center, y_center, width, height, class_idx])

        return boxes


def test_dataset_initialization():
    dataset = VOCDataset(
        img_dir="data/VOC/VOCdevkit/VOC2012/JPEGImages",
        label_dir="data/VOC/VOCdevkit/VOC2012/Annotations",
        img_size=IMAGE_SIZE,
        anchors=ANCHORS,
        transform=train_transforms,
    )
    print("Dataset initialized successfully.")
    print(f"Number of images: {len(dataset)}")


def test_bounding_box_conversion(index=0):
    dataset = VOCDataset(
        img_dir="data/VOC/VOCdevkit/VOC2012/JPEGImages",
        label_dir="data/VOC/VOCdevkit/VOC2012/Annotations",
        img_size=IMAGE_SIZE,
        anchors=ANCHORS,
        transform=train_transforms,
    )
    image, targets = dataset[index]
    print("\nTesting bounding box conversion...")
    print(f"Image shape: {image.shape}")
    print(f"Targets (for each scale):")
    for scale, target in enumerate(targets):
        print(f"Scale {scale + 1}: Target shape: {target.shape}")
        print(target)


def test_iou_calculation():
    dataset = VOCDataset(
        img_dir="data/VOC/VOCdevkit/VOC2012/JPEGImages",
        label_dir="data/VOC/VOCdevkit/VOC2012/Annotations",
        img_size=IMAGE_SIZE,
        anchors=ANCHORS,
        transform=train_transforms,
    )

    # Convert anchors to corner format for intersection_over_union calculation
    anchors_corner_format = []
    for anchor in dataset.anchors:
        x_center, y_center = (
            0.0,
            0.0,
        )  # assuming anchors are centered at origin
        width, height = anchor[0], anchor[1]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        anchors_corner_format.append([x1, y1, x2, y2])

    # Sample box in corners format
    sample_box = [0.3, 0.2, 0.7, 0.8]  # x1, y1, x2, y2 format
    anchors_corner_format = torch.tensor(anchors_corner_format)
    sample_box = torch.tensor(sample_box)

    print("\nTesting IoU calculation with intersection_over_union function...")
    print(f"Sample box (corners format): {sample_box}")
    print(f"Anchors (corners format): {anchors_corner_format}")
    iou_scores = torch.tensor(
        [
            intersection_over_union(sample_box, anchor)
            for anchor in anchors_corner_format
        ]
    )
    print(f"IoU scores with anchors: {iou_scores}")


# Run tests
if __name__ == "__main__":
    download_voc_dataset(year="2012", root="data/VOC")
    test_dataset_initialization()
    test_bounding_box_conversion()
    test_iou_calculation()
