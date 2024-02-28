import torch.nn as nn
from torchinfo import summary
from helper import (
    ResidualBlock,
    NonLocalBlock,
    UpSampleBlock,
    GroupNorm,
    Swish,
)
import config


class Decoder(nn.Module):
    def __init__(self, config):
        """
        Decoder module for generating images from a latent space.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        super(Decoder, self).__init__()
        self.config = config

        channels = config.decoder_channels
        attn_resolution = config.attn_resolution
        num_res_blocks = config.decoder_num_res_blocks
        resolution = config.decoder_resolution

        in_channels = channels[0]
        layers = [
            nn.Conv2d(config.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i, out_channels in enumerate(channels):
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, config.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape
            (batch_size, latent_dim, height, width).

        Returns:
            torch.Tensor: Output tensor of shape
            (batch_size, image_channels, height, width).
        """
        return self.model(x)


def test_decoder():
    """
    Test function to create and summarize the decoder model.

    This function creates an instance of the Decoder module using the
    configuration from DECODER_CONFIG and prints the model architecture
    summary.
    """
    # Create an instance of the Decoder module using the configuration
    decoder = Decoder(config)

    # Print the decoder model to display its architecture
    print("Decoder Model:")
    print(decoder)

    # Generate a summary of the decoder model
    print("\nDecoder Model Summary:")
    summary(decoder)


if __name__ == "__main__":
    test_decoder()
