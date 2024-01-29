# Vision Transformer (ViT) Implementation in PyTorch

This repository contains an implementation of the Vision Transformer (ViT) model in PyTorch. ViT is a transformer-based architecture introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al. It achieves state-of-the-art results on various image classification tasks by applying the transformer architecture directly to images.

## Paper Overview

The Vision Transformer (ViT) model consists of the following key components:

- **Patch Embeddings**: The input image is divided into fixed-size patches, which are linearly embedded to generate token embeddings.
- **Positional Encodings**: Positional encodings are added to the token embeddings to preserve spatial information.
- **Transformer Encoder**: A transformer encoder processes the token embeddings to capture global and local dependencies.
- **Classification Head**: A simple classification head (e.g., linear layer) is added on top of the transformer encoder to predict class labels.

### Architecture Formulae

The core ViT architecture can be summarized with the following formulae:

1. **Patch Embeddings**:

   ![Patch Embeddings Formula](https://latex.codecogs.com/svg.latex?X%20%3D%20%5Ctext%7BPatchEmbeddings%7D(images))

2. **Positional Encodings**:

   ![Positional Encodings Formula](https://latex.codecogs.com/svg.latex?X%20%3D%20X%20%2B%20%5Ctext%7BPositionalEncodings%7D(X))

3. **Transformer Encoder**:

   ![Transformer Encoder Formula](https://latex.codecogs.com/svg.latex?X%20%3D%20%5Ctext%7BTransformerEncoder%7D(X))

4. **Classification Head**:

   ![Classification Head Formula](https://latex.codecogs.com/svg.latex?output%20%3D%20%5Ctext%7BClassificationHead%7D(X))

## Code Implementation

The `vit.py` script in this repository contains the implementation of the Vision Transformer model in PyTorch. It includes classes for PatchEmbeddings, PositionalEncoding, TransformerEncoder, and a simple classification head.

### Additional Features

- **Pre-trained Models**: We provide pre-trained ViT models that can be directly used for transfer learning on downstream tasks.
- **Data Augmentation**: The code includes data augmentation techniques such as random cropping, random flipping, and color jittering to enhance model robustness.
- **Evaluation Metrics**: Evaluate the model performance using common evaluation metrics such as accuracy, precision, recall, and F1-score.

Feel free to explore the code, experiment with different hyperparameters, and adapt it for your own image classification tasks!

## Usage

To use the ViT model in your own projects, simply import the relevant classes from `vit.py` and instantiate the model. You can then train the model on your dataset using standard PyTorch training procedures.

```python
from vit import VisionTransformer

# Instantiate ViT model
model = ViT(image_size=224, patch_size=16, num_classes=1000, num_layers=12, dim=768)

# Train the model
# ...
