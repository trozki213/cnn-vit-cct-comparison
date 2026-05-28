# models/vit_small_script.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ContigGrad(torch.autograd.Function):
    """Identity in forward; makes the gradient contiguous in backward.
    Inserted before view-based ops (flatten, view) so their backward,
    which internally uses view(), receives a contiguous gradient even
    when upstream ops (e.g. torch.cat) produce non-contiguous slices."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.contiguous()


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ContigGrad.apply(self.projection(x).flatten(2)).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, E = x.shape
        H, D = self.attn.num_heads, self.attn.head_dim
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        attn = attn.softmax(dim=-1)
        self.attn._captured = attn.detach().cpu()
        if self.training and self.attn.dropout > 0:
            attn = F.dropout(attn, p=self.attn.dropout)
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        return F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
            ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViT_small(nn.Module):
    """
    Vanilla Vision Transformer for CIFAR-10 (~0.73M params).

    Target: ~0.73M parameters on CIFAR-10 (num_classes=10)
    """

    def __init__(
        self,
        ch: int         = 3,
        img_size: int   = 32,
        patch_size: int = 4,
        emb_dim: int    = 128,
        n_layers: int   = 5,
        heads: int      = 4,
        mlp_dim: int    = 320,
        dropout: float  = 0.1,
        out_dim: int    = 10,
    ) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.patch_embedding = PatchEmbedding(ch, patch_size, emb_dim)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout       = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return self.head(x[:, 0, :])


if __name__ == "__main__":
    model = ViT_small()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # expected: ~0.73M

    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")           # expected: torch.Size([4, 10])
