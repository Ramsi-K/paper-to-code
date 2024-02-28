import torch.nn as nn
from torchsummary import summary
from helper import (
    ResidualBlock,
    NonLocalBlock,
    UpSampleBlock,
    GroupNorm,
    Swish,
)


class Decoder(nn.Module):
    def __init__(
        self,
        *args,
        # latent_dim,
        # image_channels,
    ):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolution = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i, out_channels in enumerate(channels):
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(UpSampleBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main():
    latent_dim = 512
    image_channels = 3
    decoder = Decoder(latent_dim, image_channels)
    print(decoder)


if __name__ == "__main__":
    main()
