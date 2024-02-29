import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class ImagePaths(Dataset):
    """
    Dataset class to load and preprocess images from a directory.

    Args:
        path (str): Path to the directory containing images.
        size (int, optional): Desired size for resizing images. Defaults to None.

    Attributes:
        size (int): Desired size for resizing images.
        images (list): List of image file paths.
        _length (int): Number of images in the dataset.
        rescaler (albumentations.SmallestMaxSize): Rescaling transformation.
        cropper (albumentations.CenterCrop): Cropping transformation.
        preprocessor (albumentations.Compose): Composition of preprocessing transformations.
    """

    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(
            height=self.size, width=self.size
        )
        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper]
        )

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        """
        Preprocesses an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image.
        """
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    """
    Loads data using DataLoader.

    Args:
        args (Namespace): Configuration parameters.

    Returns:
        DataLoader: DataLoader object containing the loaded data.
    """
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    return train_loader


def weights_init(m):
    """
    Initializes weights of Conv and BatchNorm layers.

    Args:
        m (nn.Module): Neural network module.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    """
    Plots input, reconstructed, half-sampled, and fully sampled images.

    Args:
        images (dict): Dictionary containing input, reconstructed, half-sampled, and fully sampled images.
    """
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(
        reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0)
    )
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
