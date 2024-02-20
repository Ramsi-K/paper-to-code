# ProGAN Implementation

## Summary

This repository contains an implementation of the Progressive Growing of GANs (ProGAN) as described in the paper "Progressive Growing of GANs for Improved Quality, Stability, and Variation" by Karras et al. ProGAN is a deep learning architecture that gradually increases the resolution of both the generator and discriminator during training, leading to high-quality and diverse image generation.

## Discussion

Traditional GANs often struggle to generate high-quality images at large scales due to training instability and mode collapse. ProGAN addresses these challenges by gradually increasing both the resolution of generated images and the complexity of the network architecture during training. This progressive growth strategy enables the model to learn finer details progressively, resulting in higher-resolution and more realistic images.

At the core of ProGAN is its innovative architecture, which consists of a series of convolutional neural networks (CNNs) organized in a hierarchical manner. Unlike conventional GANs, where the generator produces low-resolution images that are subsequently upscaled, ProGAN generates images directly at the target resolution from the outset. Moreover, ProGAN introduces a novel training regime that involves alternating between phases of resolution growth and stabilization, allowing the model to learn progressively more complex features.

One of the key contributions of ProGAN is the concept of minibatch standard deviation, which enhances the diversity and quality of generated images by encouraging the generator to produce varied outputs. Additionally, ProGAN employs spectral normalization in both the generator and discriminator (referred to as the "critic" in ProGAN) to stabilize training and prevent mode collapse. Through extensive experimentation on a variety of datasets, including faces and bedrooms, ProGAN demonstrates superior performance compared to existing state-of-the-art methods, producing high-fidelity images with remarkable visual quality and diversity.

## Methodology

To convert the ProGAN paper to code, follow these steps:

1. **Model Architecture**: Design the generator and discriminator networks following the progressive growing strategy outlined in the paper.
2. **Progressive Training**: Implement the progressive growing training procedure, where the model is gradually trained on low-resolution images before transitioning to higher resolutions.
3. **Training Stability**: Incorporate techniques such as minibatch standard deviation and pixelwise feature vector normalization to improve training stability and convergence.
4. **Evaluation**: Evaluate the performance of the trained model using qualitative metrics such as image quality, diversity, and FID score.

## Repository Structure

- `train.py`: Python script for training the ProGAN model.
- `model.py`: Python module containing the implementation of the ProGAN architecture.
- `utils.py`: Utility functions for data loading, visualization, and evaluation.
- `config.py`: Configuration file for specifying hyperparameters and training settings.
- `README.md`: This file, providing an overview of the repository and instructions for usage.

## Dataset

The dataset was sourced from Kaggle.

Link: [CelebA-HQ](https://www.kaggle.com/datasets/lamsimon/celebahq)

Train_Male: 10057 images
Train_Female: 17943 images

## Usage

1. Clone the repository: `https://github.com/Ramsi-K/paper-to-code.git`
2. Install dependencies: `conda env create --name <env_name> -f environment.yml`
3. Adjust hyperparameters and configurations in the `config.py` file if necessary.
4. Prepare the dataset and place it in the appropriate directory.
5. Run the training script: `python train.py`

## References

- [Paper: Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [Blog: 'ProGAN: How NVIDIA Generated Images of Unprecedented Quality'](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2)
- [Official Tensorflow Implementation](https://github.com/tkarras/progressive_growing_of_gans?tab=readme-ov-file)
- [Unofficial Pytorch Implementation](https://github.com/akanimax/pro_gan_pytorch)
- [Aladdin Persson- ProGAN paper walkthrough video](https://www.youtube.com/watch?v=lhs78if-E7E&ab_channel=AladdinPersson)
- [Aladdin Persson - GitHub](https://github.com/aladdinpersson)