# YOLOv3 Implementation for Object Detection

This repository contains an implementation of the YOLOv3 (You Only Look Once, version 3) architecture in PyTorch, using the Pascal VOC dataset for training and inference. YOLOv3 is known for its real-time object detection capability and high accuracy, making it suitable for applications requiring fast processing.

## Key Features

- **Multi-scale predictions**: YOLOv3 performs detection at three different scales, capturing objects of varying sizes effectively.
- **Bounding box regression with anchor boxes**: Leverages predefined anchor boxes to improve detection accuracy and localization.
- **Efficient model architecture**: Built on a modified Darknet-53 backbone, providing a balance of speed and accuracy.

## Purpose and Future Work

My objective with this project is to gain a deep understanding of the YOLO family of models, starting with YOLOv3. I am working toward training a detection model on the LUNA16 dataset, focusing on lung nodule detection. My intention is to apply this knowledge progressively, by using YOLO on 2D slices, I aim to examine the model’s effectiveness in detecting nodules or similar features within individual slices, which could later inform the development of a more comprehensive 3D detection pipeline.

While actual training on the LUNA16 dataset will be done using latest version of YOLO, this project serves as a foundational exercise to understand the evolution of YOLO architectures and the theoretical advancements that each version brings.

## Methodology

This implementation follows a step-by-step approach based on the YOLOv3 paper:

1. **Architecture Understanding**: Gaining familiarity with the YOLOv3 architecture, including the Darknet-53 backbone and the multi-scale prediction mechanism.
2. **Anchor Boxes & Bounding Boxes**: Configuring anchor boxes and bounding boxes based on the dataset.
3. **Model Implementation**: Constructing the YOLOv3 network in PyTorch, adhering closely to the structural details described in the paper.
4. **Loss Function Design**: Implementing the custom YOLOv3 loss function, which combines localization, confidence, and class prediction losses.
5. **Training Strategy**: Training the network on a dataset with annotated bounding boxes and class labels, focusing on maintaining a balance between speed and accuracy.

## Theory: Understanding YOLOv3

### Architecture

YOLOv3 is a single-stage object detector, meaning it performs object classification and localization in a single pass through the network. The model is built on the Darknet-53 backbone, which consists of 53 convolutional layers with residual connections. YOLOv3 adds 53 more layers on top of this backbone for a total of 106 layers. These additional layers are responsible for making predictions at three different scales to detect objects of varying sizes.

### Metrics

YOLOv3 uses Intersection over Union (IoU) and mean Average Precision (mAP) as performance metrics:

- **IoU (Intersection over Union)**: Measures the overlap between the predicted bounding box and the ground truth bounding box. A higher IoU indicates better localization accuracy.
- **mAP (mean Average Precision)**: The primary metric for object detection, mAP is calculated by averaging the precision across different IoU thresholds. It reflects both the accuracy of the bounding box predictions and the model's confidence in those predictions.

## Project Structure

YOLOv3/  
├── config.py                # Configuration file with parameters and settings  
├── dataset.py               # Dataset handling and preprocessing  
├── model.py                 # YOLOv3 model implementation  
├── train.py                 # Training loop  
├── inference.py             # Inference script for testing model predictions  
├── utils.py                 # Utility functions (NMS, plotting, logging)  
├── requirements.txt         # Required packages for the project  
└── results/                 # Directory for saving inference results and logs  

## Key Learnings and Challenges

1. **Class Prediction Adjustments**  
   In YOLOv3, we initially used raw class probabilities for calculating the loss. Through iterative debugging, it became clear that these full probability arrays were unnecessary for calculating class labels. We optimized by selecting the class with the highest confidence, simplifying our loss input while retaining accuracy. This insight reduced dimensionality without impacting model performance.

2. **Handling Class Boundaries and Label Constraints**  
   Encountering “IndexError: Target out of bounds” highlighted the need to explicitly ensure class indices align with the dataset. This led to adding assertions that confirmed all labels remained within valid bounds, given our 20 classes for Pascal VOC. This check prevented runtime crashes, improving robustness.

3. **Formatted Target Consistency Across Scales**  
   The varying grid sizes across prediction scales (small, medium, large) required careful attention to ensure target and prediction formats aligned across scales. Each scale demanded its own formatted target, structured to match the corresponding prediction shape. Implementing consistent formatting improved alignment and reduced errors related to mismatched tensors.

4. **Objectness and Non-Object Masking**  
   Efficiently separating object and non-object areas in the target required generating specific masks. Utilizing binary masks for object presence (object mask) and absence (no-object mask) allowed us to precisely calculate object loss and no-object loss. This approach ensured that the model learned to prioritize regions of interest effectively.

5. **Anchor Tensor Management**  
   Anchors needed resizing and reshaping to match predictions across scales. Standardizing anchors improved bounding box precision, leading to more accurate object localization. This structure also allowed the anchor tensors to be used consistently without further adjustments, simplifying our code.

6. **Iterative Debugging of Model Shapes and Tensor Dimensions**  
   Throughout, maintaining the expected shapes of predictions and targets was essential for avoiding mismatches. Regular print statements and shape checks were invaluable in tracking these dimensions, ensuring loss calculations ran smoothly across all scales and reducing shape mismatch errors. This practice evolved into a routine check, saving time in resolving tensor issues.

7. **Loss Component Isolation and Verification**  
   Debugging each loss component independently allowed for isolated testing of object loss, no-object loss, box loss, and class loss. This approach revealed specific areas for improvement, particularly in the class loss, which initially failed due to label mismatches. Testing each component separately made troubleshooting more efficient.

## Setup and Installation

1. **Clone the repository**:  

    ``` bash  
    git clone <https://github.com/yourusername/YOLOv3>  
    cd YOLOv3  
    ```

2. **Install dependencies**:  

    ``` bash  
    pip install -r requirements.txt  
    ```

3. **Download Pascal VOC Dataset** (used for training and inference):  
   The dataset will automatically download when you run `train.py`. It is stored in the `data/VOC` directory by default.

## Usage

### Training

To train the YOLOv3 model, run the following command:  

``` bash  
python train.py  
```

This script will download the Pascal VOC dataset if it is not already present and begin training the YOLOv3 model. Model checkpoints will be saved every 5 epochs.

### Inference

To perform inference on test images, use the `inference.py` script:  

``` bash  
python inference.py  
```

Inference results (annotated images) and logs will be saved in the `results/` directory.

## Configuration

Adjust settings in `config.py` to customize parameters like learning rate, batch size, or directory paths. Update `IMG_DIR` and `LABEL_DIR` in `config.py` if using a different dataset.

## Troubleshooting

- **Dataset Issues**: Ensure the Pascal VOC dataset is correctly downloaded in the `data/VOC` directory.
- **CUDA Errors**: Verify your PyTorch installation supports GPU acceleration, or set `DEVICE` to `"cpu"` in `config.py`.
- **Import Errors**: Check that all dependencies in `requirements.txt` are installed.

## References

- Original YOLOv3 Paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Story of anchors and grids in YOLO - Onkar Chougule](https://medium.com/@onkar.cho1/story-of-anchors-and-grids-in-yolo-68efb7a1a04b#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImU4NjNmZTI5MmZhMmEyOTY3Y2Q3NTUxYzQyYTEyMTFiY2FjNTUwNzEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTc5NzgyMjk2OTM1NTA0NjIzOTYiLCJlbWFpbCI6InJhbXNpLmthbGlhQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3MzA4OTU3NjEsIm5hbWUiOiJSYW1zaSBLYWxpYSIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJaGpJaFVHN1hMQmlRSHRRMGZuMThiY0I0ZFROa0NDY3NDOXdDdllWUDhSN29ZZ1E9czk2LWMiLCJnaXZlbl9uYW1lIjoiUmFtc2kiLCJmYW1pbHlfbmFtZSI6IkthbGlhIiwiaWF0IjoxNzMwODk2MDYxLCJleHAiOjE3MzA4OTk2NjEsImp0aSI6IjQ2Yjg1MGVjZThmM2JlMmUzY2I2ODk1MmY0OTY1MDliY2VjNDQyNzAifQ.L4WnQMovNCdoCb8HDdfKlOcPQGiyi-_bAbXIZOSii2v6QT63-k5CtsfTBmooNYMcvOug_haROYLoi_kF40KUfQpfHHQMD74GnIfAQRIR7VkXljw7zcBMbCe5gjmy0Is7YjbkVi0LHkR_8T3KKfEy97m-tT3QFYd24wwzH54pfRLwHpHS1x4mh3m_7NlZ-l_-5lJ6LRzaKa_WM7SKBLnNAZ5MDxshG-d8DgH02UXJnAWI0ghahimYEIME643hAFSOUrjFANQ5mALP7enjSFmw7tOYUo_Huehgl-LpSY24b5V-0pc195XIK0qdmhJRKuqVqKal0DwWkaDjhXQIcK35KQ)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
