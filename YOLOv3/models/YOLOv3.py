# models/YOLOv3.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A convolutional block that consists of Conv2D -> BatchNorm -> LeakyReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  # Maintain spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.layer1 = ConvBlock(
            channels, channels // 2, kernel_size=1, stride=1
        )
        self.layer2 = ConvBlock(
            channels // 2, channels, kernel_size=3, stride=1
        )

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        return x + residual


class Darknet53(nn.Module):
    """
    The backbone network for YOLOv3, consisting of convolutional and residual layers.
    """

    def __init__(self):
        super(Darknet53, self).__init__()
        self.initial_conv = ConvBlock(3, 32, kernel_size=3, stride=1)
        self.layers = nn.ModuleList()
        # (out_channels, num_blocks)
        layer_configs = [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)]
        in_channels = 32
        for out_channels, num_blocks in layer_configs:
            self.layers.append(
                self._make_layer(in_channels, out_channels, num_blocks)
            )
            in_channels = out_channels

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        x = self.initial_conv(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in [
                2,
                3,
                4,
            ]:  # Collect outputs at scales 52x52, 26x26, 13x13
                outputs.append(x)
        return outputs  # Returns feature maps at different scales


class YOLOv3(nn.Module):
    """
    YOLOv3 model that utilizes Darknet53 backbone and predicts bounding boxes at three scales.
    """

    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.backbone = Darknet53()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Define convolutional layers for detection at each scale
        self.conv_set1, self.det_conv1 = self._make_detection_layers(1024, 512)
        # Adjusted in_channels to 512 to match the concatenated feature map size
        self.conv_set2, self.det_conv2 = self._make_detection_layers(
            512 + 512, 256
        )
        self.conv_set3, self.det_conv3 = self._make_detection_layers(
            256 + 256, 128
        )

    def _make_detection_layers(self, in_channels, out_channels):
        """
        Creates a set of convolutional layers for detection at a given scale.
        """
        conv_set = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, 1),
            ConvBlock(out_channels, out_channels * 2, 3, 1),
            ConvBlock(out_channels * 2, out_channels, 1, 1),
            ConvBlock(out_channels, out_channels * 2, 3, 1),
            ConvBlock(out_channels * 2, out_channels, 1, 1),
        )
        det_conv = nn.Sequential(
            ConvBlock(out_channels, out_channels * 2, 3, 1),
            nn.Conv2d(
                out_channels * 2,
                3 * (self.num_classes + 5),
                kernel_size=1,
                stride=1,
            ),
        )
        return conv_set, det_conv

    def forward(self, x):
        # Feature maps from the backbone
        features = self.backbone(x)
        outputs = []

        # Detection at the largest scale (13x13)
        x = self.conv_set1(features[2])
        out1 = self.det_conv1(x)
        outputs.append(out1)

        # Prepare for the medium scale detection (26x26)
        x = self.upsample(x)
        x = torch.cat(
            [x, features[1]], dim=1
        )  # Concatenate with feature map from backbone
        x = self.conv_set2(x)
        out2 = self.det_conv2(x)
        outputs.append(out2)

        # Prepare for the smallest scale detection (52x52)
        x = self.upsample(x)
        x = torch.cat(
            [x, features[0]], dim=1
        )  # Concatenate with feature map from backbone
        x = self.conv_set3(x)
        out3 = self.det_conv3(x)
        outputs.append(out3)

        return outputs  # List of outputs at different scales
