import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """
    Group Normalization layer.

    Args:
        channels (int): Number of channels in the input tensor.

    Returns:
        torch.Tensor: Output tensor after applying Group Normalization.
    """

    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    """
    Swish activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying Swish activation.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and Swish activations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        torch.Tensor: Output tensor after applying the residual block.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
        )

        # Add a 1x1 convolution if the number of input
        # channels is different from output channels
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, x):
        # Apply the residual block
        if self.in_channels != self.out_channels:
            return self.channel_up(x) * self.block(x)
        else:
            return x * self.block(x)


class UpSampleBlock(nn.Module):
    """
    Up-sampling block using interpolation.

    Args:
        channels (int): Number of input channels.

    Returns:
        torch.Tensor: Output tensor after up-sampling.
    """

    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # Up-sample using interpolation and apply convolution
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    """
    Down-sampling block using convolution.

    Args:
        channels (int): Number of input channels.

    Returns:
        torch.Tensor: Output tensor after down-sampling.
    """

    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        # Down-sample using convolution
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """
    Non-local block for capturing long-range dependencies.

    Args:
        channels (int): Number of input channels.

    Returns:
        torch.Tensor: Output tensor after applying the non-local block.
    """

    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Apply GroupNorm
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for matrix multiplication
        b, c, h, w = q.shape
        q = q.view(b, c, h * w).permute(0, 2, 1)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute attention map
        attn = torch.bmm(q, k) * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2).permute(0, 2, 1)
        A = torch.bmm(v, attn).view(b, c, h, w)
        A = self.proj_out(A)

        return x * A
