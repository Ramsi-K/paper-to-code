import torch.nn as nn
from torchvision.models import vgg19
import config


class VGGLoss(nn.Module):
    """
    VGG Loss module for perceptual loss calculation using VGG19 feature extractor.

    Args:
        None

    Returns:
        None
    """

    def __init__(self) -> None:
        """
        Initialize the VGGLoss module.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        # Load pre-trained VGG19 model and extract relevant features
        self.vgg = (
            vgg19(pretrained=True).features[:35].eval().to(config.DEVICE)
        )

        # Set requires_grad to False for all parameters in VGG model
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define Mean Squared Error (MSE) loss function
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        """
        Forward pass of the VGGLoss module.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The VGG loss value.
        """
        # Get VGG features of input and target tensors
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)

        # Compute MSE loss between VGG features of input and target
        return self.loss(vgg_input_features, vgg_target_features)
