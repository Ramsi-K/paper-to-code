import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

    # def test():
    #     x = torch.randn(5, 3, 256, 256)
    #     model = Discriminator(in_channels=3)
    #     preds = model(x)
    #     print(f"Preds shape: {preds.shape}")

    # if __name__ == "__main__":
    #     test()

    import torch


import torch.nn as nn

# Your classes here...


def test_discriminator():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the discriminator
    discriminator = Discriminator().to(device)

    # Create dummy input data
    batch_size = 4
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, 3, height, width).to(device)

    # Perform a forward pass
    output = discriminator(dummy_input)

    # Check the output shape
    expected_output_shape = (batch_size, 1, height // 16, width // 16)
    assert (
        output.shape == expected_output_shape
    ), f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"

    print("Discriminator test passed!")


# Run the test function
test_discriminator()
