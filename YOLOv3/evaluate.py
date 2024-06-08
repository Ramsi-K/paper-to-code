# evaluate.py

import torch
from torch.utils.data import DataLoader
from models.YOLOv3 import YOLOv3
from utils.utils import non_max_suppression
from dataset import VOCDataset
import config


def evaluate():
    """
    Evaluate the YOLOv3 model on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH))
    model.eval()

    test_dataset = VOCDataset(
        config.DATA_DIR, split="test", img_size=config.IMG_SIZE
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Perform Non-Max Suppression
            predictions = non_max_suppression(
                outputs, config.CONF_THRESHOLD, config.NMS_IOU_THRESHOLD
            )
            # Process predictions (visualization or metric computation)
            # Placeholder for mAP calculation
            pass


if __name__ == "__main__":
    evaluate()
