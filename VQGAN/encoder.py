import torch.nn as nn
from torchinfo import summary
from helper import (
    ResidualBlock,
    NonLocalBlock,
    DownSampleBlock,
    GroupNorm,
    Swish,
)
import config  # Importing encoder configuration from config.py


class Encoder(nn.Module):
    """
    Encoder module for encoding input images into a latent space.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Attributes:
        config (dict): Configuration parameters for the encoder.
        model (nn.Module): Sequential model representing the
        encoder architecture.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        channels = config.encoder_channels
        num_res_block = config.encoder_num_res_block
        resolution = config.encoder_resolution
        image_channels = config.image_channels
        latent_dim = config.latent_dim

        # Initialize the layers list with the initial convolutional layer
        layers = [
            nn.Conv2d(
                image_channels, channels[0], kernel_size=3, stride=1, padding=1
            )
        ]

        # Iterate over the channels to create the residual
        # blocks and downsampling layers
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            # Create multiple residual blocks for each channel transition
            for _ in range(num_res_block):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                # Add non-local attention block if
                # required by the configuration
                if resolution in config.attn_resolution:
                    layers.append(NonLocalBlock(in_channels))

            # Add downsampling layer if it's not the last channel transition
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(out_channels))
                resolution //= 2

        # Add final layers including residual blocks,
        # normalization, activation, and convolution
        layers += [
            ResidualBlock(channels[-1], channels[-1]),
            NonLocalBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
            GroupNorm(channels[-1]),
            Swish(),
            nn.Conv2d(
                channels[-1], latent_dim, kernel_size=3, stride=1, padding=1
            ),
        ]

        # Define the model as a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape
            (batch_size, image_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape
            (batch_size, latent_dim, height, width).
        """
        return self.model(x)


def test_encoder():
    """
    Test function to create and summarize the encoder model.

    This function creates an instance of the Encoder module using the
    configuration from ENCODER_CONFIG and prints the model architecture
    summary.
    """
    # Create an instance of the Encoder module using the configuration
    encoder = Encoder(config)

    # Print information about the encoder configuration
    print("Encoder Configuration:")
    print(config)

    # Print a separator for better readability
    print("\n" + "=" * 50 + "\n")

    # Print the encoder model to display its architecture
    print("Encoder Model:")
    print(encoder)

    # Generate a summary of the encoder model
    print("\nEncoder Model Summary:")
    print_encoder_summary(encoder)


def print_encoder_summary(encoder):
    """
    Print the summary of the encoder model.

    Args:
        encoder (Encoder): Instance of the Encoder module.
    """
    summary(encoder)


if __name__ == "__main__":
    test_encoder()
