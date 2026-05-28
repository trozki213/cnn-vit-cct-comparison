# models/cct_large_script.py

import torch
import torch.nn as nn


class CCT_large(nn.Module):
    """
    Compact Convolutional Transformer - CCT-7/3x2 custom (~5.2M)
    Hassani et al., 2021 - "Escaping the Big Data Paradigm with Compact Transformers"

    Target: ~5M parameters on CIFAR-10 (num_classes=10)

    Key differences from vanilla ViT:
      - Convolutional tokenizer: extracts local features before attention (locality bias)
      - Sequence pooling: replaces CLS token with a learned weighted average over all patches
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim:   int = 300,
        num_heads:   int = 4,
        num_layers:  int = 7,
        mlp_dim:     int = 600,     # MLP/embed ratio = 2, as in paper for CCT-7
        dropout:     float = 0.1,
    ):
        super().__init__()

        # --- Convolutional Tokenizer ---
        # Two conv blocks with MaxPool each halve spatial resolution: 32 -> 16 -> 8
        # Output: (B, embed_dim, 8, 8) = 64 patch tokens
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # --- Positional Embedding ---
        # Learnable position encoding for the 64 patch tokens (no CLS token)
        num_patches = 8 * 8
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        # Pre-LN (norm_first=True) for better stability in low-data regimes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # --- Sequence Pooling ---
        # Learns a scalar attention score per patch, then computes a weighted sum
        # More expressive than a fixed CLS token
        self.attention_pool = nn.Linear(embed_dim, 1)

        # --- Classification Head ---
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional tokenization: image -> patch token sequence
        x = self.tokenizer(x)               # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)    # (B, 64, embed_dim)

        # Add positional information and apply dropout
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Self-attention over the patch sequence
        x = self.transformer(x)             # (B, 64, embed_dim)
        x = self.norm(x)

        # Sequence pooling: weighted average over all patch tokens
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)  # (B, 64, 1)
        x = (attn_weights * x).sum(dim=1)                            # (B, embed_dim)

        # Classification
        return self.head(x)


if __name__ == "__main__":
    model = CCT_large(num_classes=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # expected: ~5.2M

    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")           # expected: torch.Size([4, 10])
