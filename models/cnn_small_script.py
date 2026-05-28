# models/cnn_small_script.py

import torch
import torch.nn as nn


class CNN_small(nn.Module):
    """
    Small CNN for CIFAR-10 (~0.77M params).

    Target: ~0.75M parameters on CIFAR-10 (num_classes=10)

    Three convolutional stages progressively double channels (64 -> 128 -> 256)
    while halving spatial resolution via MaxPool (32 -> 16 -> 8 -> 4). The
    flattened feature map (256*4*4 = 4096) feeds a two-layer head with a
    bottleneck at 96 units and Dropout for regularisation.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: (B, 3, 32, 32) -> (B, 64, 16, 16)
            nn.Conv2d(3,   64,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (B, 64, 16, 16) -> (B, 128, 8, 8)
            nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (B, 128, 8, 8) -> (B, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNN_small(num_classes=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # expected: ~0.77M

    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")           # expected: torch.Size([4, 10])
