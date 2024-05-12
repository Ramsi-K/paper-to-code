import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from PIL import Image, ImageFile
import xml.etree.ElementTree as ET
from config import (
    IMG_DIR,
    LABEL_DIR,
    train_transforms,
    test_transforms,
    PASCAL_CLASSES,
)
from utils import iou_width_height as iou

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class YOLODataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        anchors,
        S=[13, 26, 52],
        C=20,  # Limit classes to 20 as set in config
        transform=None,
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_filename = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = image.size
        image = np.array(image)

        # Replace the .jpg extension with .xml to find the corresponding label file
        label_filename = img_filename.replace(".jpg", ".xml")
        label_path = os.path.join(self.label_dir, label_filename)
        bboxes = self._parse_annotations(label_path, image_width, image_height)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [
            torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S
        ]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][
                        anchor_on_scale, i, j, 1:5
                    ] = box_coordinates
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
        """
        Parses the .xml label file and returns bounding boxes in YOLO format.
        """
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            class_idx = PASCAL_CLASSES.index(label)
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


if __name__ == "__main__":
    download_voc_dataset(year="2012", root="data/VOC")
    print("Dataset download and setup complete.")


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision.datasets import VOCDetection
# from torchvision.transforms import ToTensor
# from PIL import Image, ImageFile
# import xml.etree.ElementTree as ET
# from config import (
#     IMG_DIR,
#     LABEL_DIR,
#     train_transforms,
#     test_transforms,
#     PASCAL_CLASSES,
# )
# from utils import iou_width_height as iou

# ImageFile.LOAD_TRUNCATED_IMAGES = True


# def download_voc_dataset(year="2012", root="data/VOC"):
#     if os.path.exists(os.path.join(root, "VOCdevkit", f"VOC{year}")):
#         print(
#             f"Pascal VOC {year} dataset already exists in {root}. Skipping download."
#         )
#     else:
#         VOCDetection(
#             root=root,
#             year=year,
#             image_set="train",
#             download=True,
#             transform=train_transforms,
#         )
#         print(f"Pascal VOC {year} dataset downloaded successfully.")


# class YOLODataset(Dataset):
#     def __init__(
#         self,
#         img_dir,
#         label_dir,
#         anchors,
#         S=[13, 26, 52],
#         C=20,  # Limit classes to 20 as set in config
#         transform=None,
#     ) -> None:
#         super().__init__()
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.S = S
#         self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
#         self.num_anchors = self.anchors.shape[0]
#         self.num_anchors_per_scale = self.num_anchors // 3
#         self.C = C
#         self.ignore_iou_thresh = 0.5
#         self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, index):
#         img_filename = self.img_files[index]
#         img_path = os.path.join(self.img_dir, img_filename)
#         image = Image.open(img_path).convert("RGB")
#         image_width, image_height = image.size
#         image = np.array(image)

#         # Replace the .jpg extension with .xml to find the corresponding label file
#         label_filename = img_filename.replace(".jpg", ".xml")
#         label_path = os.path.join(self.label_dir, label_filename)
#         bboxes = self._parse_annotations(label_path)

#         if self.transform:
#             augmentations = self.transform(image=image, bboxes=bboxes)
#             image = augmentations["image"]
#             bboxes = augmentations["bboxes"]

#         targets = [
#             torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S
#         ]
#         for box in bboxes:
#             iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
#             anchor_indices = iou_anchors.argsort(descending=True, dim=0)
#             x, y, width, height, class_label = box
#             has_anchor = [False, False, False]

#             for anchor_idx in anchor_indices:
#                 scale_idx = anchor_idx // self.num_anchors_per_scale
#                 anchor_on_scale = anchor_idx % self.num_anchors_per_scale
#                 S = self.S[scale_idx]
#                 i, j = int(S * y), int(S * x)
#                 anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

#                 if not anchor_taken and not has_anchor[scale_idx]:
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = 1
#                     x_cell, y_cell = S * x - j, S * y - i
#                     width_cell, height_cell = width * S, height * S
#                     box_coordinates = torch.tensor(
#                         [x_cell, y_cell, width_cell, height_cell]
#                     )
#                     targets[scale_idx][
#                         anchor_on_scale, i, j, 1:5
#                     ] = box_coordinates
#                     targets[scale_idx][anchor_on_scale, i, j, 5] = int(
#                         class_label
#                     )
#                     has_anchor[scale_idx] = True
#                 elif (
#                     not anchor_taken
#                     and iou_anchors[anchor_idx] > self.ignore_iou_thresh
#                 ):
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = -1
#         return image, tuple(targets)

#     def _parse_annotations(self, label_path):
#         """
#         Parses the .xml label file and returns bounding boxes in YOLO format.
#         """
#         boxes = []
#         tree = ET.parse(label_path)
#         root = tree.getroot()

#         for obj in root.findall("object"):
#             label = obj.find("name").text
#             class_idx = PASCAL_CLASSES.index(label)
#             bndbox = obj.find("bndbox")
#             xmin = int(bndbox.find("xmin").text)
#             ymin = int(bndbox.find("ymin").text)
#             xmax = int(bndbox.find("xmax").text)
#             ymax = int(bndbox.find("ymax").text)
#             x_center = (xmin + xmax) / 2 / image_width
#             y_center = (ymin + ymax) / 2 / image_height
#             width = (xmax - xmin) / image_width
#             height = (ymax - ymin) / image_height
#             boxes.append([x_center, y_center, width, height, class_idx])

#         return boxes


# if __name__ == "__main__":
#     download_voc_dataset(year="2012", root="data/VOC")
#     print("Dataset download and setup complete.")
