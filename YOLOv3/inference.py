import os
import torch
from datetime import datetime
from model import YOLOv3
from utils import load_image, non_max_suppression, cells_to_bboxes, plot_boxes
from config import DEVICE, IMG_DIR, CONF_THRESHOLD, NMS_IOU_THRESH, ANCHORS

# Directory to save results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_inference(model, image_paths):
    """
    Runs inference on images, saves and logs results.

    Parameters:
        model (YOLOv3): The trained YOLOv3 model.
        image_paths (list): List of paths to the images.

    Returns:
        None
    """
    model.eval()
    model.to(DEVICE)

    log_file = os.path.join(
        RESULTS_DIR,
        f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    with open(log_file, "w") as log:
        for image_path in image_paths:
            # Load and preprocess image
            image = load_image(image_path).to(DEVICE)
            with torch.no_grad():
                predictions = model(image)

            # Convert predictions to bounding boxes
            bboxes = []
            for i in range(3):  # 3 scales
                S = predictions[i].shape[2]
                anchors = torch.tensor(ANCHORS[i]).to(DEVICE) * S
                bboxes += cells_to_bboxes(
                    predictions[i], anchors, S=S, is_preds=True
                )

            # Apply non-max suppression to filter boxes
            bboxes = non_max_suppression(
                bboxes,
                iou_threshold=NMS_IOU_THRESH,
                threshold=CONF_THRESHOLD,
                box_format="midpoint",
            )

            # Log results
            log.write(f"Image: {image_path}\n")
            for box in bboxes:
                class_pred, confidence, x1, y1, x2, y2 = box
                log.write(
                    f"Class: {int(class_pred)}, Confidence: {confidence:.2f}, Box: ({x1}, {y1}, {x2}, {y2})\n"
                )
            log.write("\n")

            # Save annotated image
            image_name = os.path.basename(image_path).split(".")[0]
            result_image_path = os.path.join(
                RESULTS_DIR, f"{image_name}_result.png"
            )
            plot_boxes(image_path, bboxes, result_image_path)
            print(f"Saved annotated image: {result_image_path}")


def main():
    # Load model and weights
    model = YOLOv3(num_classes=20)
    model.load_state_dict(torch.load("yolov3.pth", map_location=DEVICE))
    print("Model loaded for inference.")

    # Test images (use a few images from Pascal VOC)
    test_images = [
        os.path.join(IMG_DIR, img) for img in os.listdir(IMG_DIR)[:5]
    ]

    # Run inference and save results
    run_inference(model, test_images)


if __name__ == "__main__":
    main()
