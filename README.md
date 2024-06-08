
# Paper-to-Code Implementations

## Overview

This repository is a collection of implementations of influential machine learning and deep learning papers. Each model is translated from theory to code, offering practical, executable versions of pioneering architectures. This project serves as both a learning exercise and a portfolio piece to demonstrate understanding and application of advanced models across various tasks, from generative modeling to object detection and classification.

## Repository Structure

Each model has its own directory containing the following:

- `model.py`: Model architecture.
- `train.py`: Training loop and setup.
- `inference.py`: Inference script for generating predictions.
- `utils.py`: Utility functions specific to each model.
- `README.md`: Detailed instructions and context for the implementation, including setup, usage, and results.

## Models Included

| Model            | Task                     | Dataset                 | Paper Reference | Highlights |
|------------------|--------------------------|-------------------------|-----------------|------------|
| **CycleGAN**     | Image-to-Image Translation | Cezanne2Photo; Car2CarDamage          | [CycleGAN](https://arxiv.org/abs/1703.10593) | Unpaired image translation |
| **DCGAN**        | Image Generation         | MNIST                  | [DCGAN](https://arxiv.org/abs/1511.06434) | Deep Convolutional GAN for realistic image generation |
| **ESRGAN**       | Image Super-Resolution   | Pre-trained                   | [ESRGAN](https://arxiv.org/abs/1809.00219) | Enhanced super-resolution GAN with high-quality outputs |
| **PointNet**     | 3D Object Recognition    | ModelNet40              | [PointNet](https://arxiv.org/abs/1612.00593) | Processing 3D point clouds |
| **ProGAN**       | Image Generation         | CelebA                  | [ProGAN](https://arxiv.org/abs/1710.10196) | Progressive growing GAN for high-resolution images |
| **VAE**          | Image Generation         | MNIST                   | [VAE](https://arxiv.org/abs/1312.6114) | Variational Autoencoder for learning complex distributions |
| **VQGAN**        | Image Generation         | Oxford Flowers                | [VQGAN](https://arxiv.org/abs/2012.09841) | Combines GANs with vector quantization for high-quality images |
| **ViT**          | Image Classification     | FoodVision Mini              | [ViT](https://arxiv.org/abs/2010.11929) | Vision Transformers for image recognition |
| **WGAN / WGAN-GP** | Stable GAN Training   | MNIST                  | [WGAN](https://arxiv.org/abs/1701.07875) | Improved GAN training stability |
| **YOLOv3**       | Object Detection         | Pascal VOC              | [YOLOv3](https://arxiv.org/abs/1804.02767) | Real-time object detection model |
| **pix2pix**      | Image-to-Image Translation | Sat2Map               | [pix2pix](https://arxiv.org/abs/1611.07004) | Paired image translation |
| **vgg_lpips**    | Image Similarity         | Custom                  | [LPIPS](https://arxiv.org/abs/1801.03924) | Learned Perceptual Image Patch Similarity for comparing images |

## Setup Instructions

Each model requires a specific environment configuration. Please refer to the `requirements.txt` or `environment.yml` file within each model’s directory for dependency information.

To set up a conda environment with the dependencies, use:

``` bash
conda env create -f environment.yml
conda activate paper-to-code
```

Alternatively, install dependencies via pip:

``` bash
pip install -r requirements.txt
```

## How to Use

Each model directory includes instructions for training and inference:

1. **Training**: To train a model, navigate to the corresponding directory and run:

    ``` bash
    python train.py
    ```

2. **Inference**: For inference on new data, use:

    ``` bash
    python inference.py --input_path <path_to_input> --output_path <path_to_output>
    ```

3. **Visualization**: Each model includes visualization tools to view outputs, such as generated images or bounding boxes.

## Future Directions and Potential Enhancements

1. **Documentation & Notebooks**: Adding interactive Jupyter notebooks to walk through each model’s training and inference process.
2. **Additional Models**: Expanding the repository to include:
   - **UNet** for medical image segmentation.
   - **ResNet** or **EfficientNet** as baseline image classifiers.
   - **DETR** for object detection.
   - **StyleGAN** for more sophisticated image generation.
3. **Comparison Table**: Summarizing performance metrics, training times, and notable findings in a central location to showcase the breadth of experimentation and model capabilities.

## Contributing

This repository is open for contributions. If you have suggestions or improvements, please submit a pull request or reach out with feedback.

## Contact

For any questions or feedback, please feel free to contact me via [Email](ramsi.kalia@gmail.com) or at [LinkedIn](https://www.linkedin.com/in/ramsikalia/).
