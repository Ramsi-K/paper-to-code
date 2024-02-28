"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch
import torch.nn as nn
import config

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(config.image_channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    4,
                    2 if i < n_layers else 1,
                    1,
                    bias=False,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
            ]

        layers.append(
            nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# def test_discriminator():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define configuration parameters
#     image_channels = config.image_channels

#     # Instantiate the discriminator
#     discriminator = Discriminator(config).to(device)

#     # Create dummy input data
#     batch_size = 4
#     height, width = 128, 128
#     dummy_input = torch.randn(batch_size, image_channels, height, width).to(
#         device
#     )

#     # Perform a forward pass
#     output = discriminator(dummy_input)

#     # Check the output shape
#     expected_output_shape = (batch_size, 1, height // 16, width // 16)
#     assert (
#         output.shape == expected_output_shape
#     ), f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"

#     print("Discriminator test passed!")


# # Run the test function
# test_discriminator()
