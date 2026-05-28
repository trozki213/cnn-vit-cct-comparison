# models/cnn_large_script.py

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act   = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class CNN_large(nn.Module):
    """
    Large residual CNN for CIFAR-10 (~4.9M params).

    Target: ~4.9M parameters on CIFAR-10 (num_classes=10)

    Three residual stages progressively increase channels (96 -> 192 -> 320)
    while halving spatial resolution via strided convolutions (32 -> 16 -> 8).
    Global average pooling and a linear head produce the final logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(96, 192, stride=2),
            ResidualBlock(192, 192),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(192, 320, stride=2),
            ResidualBlock(320, 320),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


if __name__ == "__main__":
    model = CNN_large(num_classes=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # expected: ~4.9M

    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")           # expected: torch.Size([4, 10])
