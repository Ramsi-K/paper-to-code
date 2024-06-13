# config.py

import os
import torch
import numpy as np

# from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paths
DATA_PATH = "data/VOC/VOCdevkit/VOC2012"  # Path to VOC dataset
ANNOTATIONS_PATH = os.path.join(DATA_PATH, "Annotations")
IMAGES_PATH = os.path.join(DATA_PATH, "JPEGImages")
IMAGE_SETS_PATH = os.path.join(DATA_PATH, "ImageSets/Main")
CHECKPOINTS_DIR = "checkpoints"
OUTPUT_DIR = "output"

# Model Parameters
IMAGE_SIZE = 416
GRID_SIZE = 13  # For original YOLOv3 architecture; changes at each scale
# ANCHORS = [
#     [(10, 13), (16, 30), (33, 23)],  # Scale for 13x13
#     [(30, 61), (62, 45), (59, 119)],  # Scale for 26x26
#     [(116, 90), (156, 198), (373, 326)],  # Scale for 52x52
# ]
ANCHORS = np.array(
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
NUM_CLASSES = 20  # VOC dataset has 20 classes
CONF_THRESHOLD = 0.5  # Confidence score threshold for filtering predictions
NMS_IOU_THRESH = 0.45  # IoU threshold for Non-Max Suppression

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
NUM_WORKERS = 4  # Adjust based on your system's capabilities

# Checkpoint and Output Settings
SAVE_CHECKPOINT = True
LOAD_CHECKPOINT = False
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_DIR, "yolov3_checkpoint.pth")
BEST_MODEL_FILE = os.path.join(CHECKPOINTS_DIR, "yolov3_best_model.pth")
LOG_INTERVAL = 10  # Log training progress every 10 batches

# Data Augmentation Settings
FLIP_PROB = 0.5
SCALE_MIN = 0.8
SCALE_MAX = 1.2


train_transforms = A.Compose(
    [
        A.RandomResizedCrop(
            IMAGE_SIZE, IMAGE_SIZE, scale=(SCALE_MIN, SCALE_MAX)
        ),
        A.HorizontalFlip(p=FLIP_PROB),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

val_transforms = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),  # Just resizing for validation/test
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

TRAIN_AUGMENT = train_transforms
VAL_AUGMENT = val_transforms


# Class Names (for VOC)
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Logging and Debugging
VERBOSE = True
