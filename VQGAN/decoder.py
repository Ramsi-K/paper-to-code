import torch.nn as nn
from torchinfo import summary
from helper import (
    ResidualBlock,
    NonLocalBlock,
    UpSampleBlock,
    GroupNorm,
    Swish,
)
from config import DECODER_CONFIG


class Decoder(nn.Module):
    def __init__(self, config):
        """
        Decoder module for generating images from a latent space.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        super(Decoder, self).__init__()
        self.config = config

        channels = config.get("channels", [512, 256, 256, 128, 128])
        attn_resolution = config.get("attn_resolution", [16])
        num_res_blocks = config.get("num_res_blocks", 3)
        resolution = config.get("resolution", 16)

        in_channels = channels[0]
        layers = [
            nn.Conv2d(config["latent_dim"], in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i, out_channels in enumerate(channels):
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

            layers += [
                ResidualBlock(in_channels, out_channels)
                for _ in range(num_res_blocks)
            ]

            if resolution in attn_resolution:
                layers.append(UpSampleBlock(out_channels))

            in_channels = out_channels

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(
            nn.Conv2d(in_channels, config["image_channels"], 3, 1, 1)
        )
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
    decoder = Decoder(DECODER_CONFIG)

    # Print the decoder model to display its architecture
    print("Decoder Model:")
    print(decoder)

    # Generate a summary of the decoder model
    print("\nDecoder Model Summary:")
    summary(decoder)


if __name__ == "__main__":
    test_decoder()
