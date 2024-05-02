"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, bn_act=True, **kwargs
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, bias=not bn_act, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in num_repeats:
            self.layers += [
                CNNBlock(channels, channels // 2, kernel_size=1, padding=0),
                CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels,
                3 * (num_classes + 5),
                bn_act=False,
                kernel_size=1,
            ),
        )
        self.num_classes=num_classes

    def forward(self,x):
        return (
            self.pred(x)
            .reshape(x.shape[0],3,self.num_classes+5,x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)



class YOLOv3(nn.Module):
    pass
