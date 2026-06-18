"""Swin Transformer (Liu et al., ICCV 2021) — student implementation.

Swin-Tiny: dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7.

Reference (read, do NOT import):
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from src.models.heads import MultiTaskHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) → (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """(num_windows*B, window_size, window_size, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # Relative position bias table: (2W-1)*(2W-1) × num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, W, W)
        coords_flat = torch.flatten(coords, 1)  # (2, W*W)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))  # (N, N)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """Single Swin Transformer block (W-MSA or SW-MSA)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor, H: int, W: int, attn_mask: Tensor | None = None) -> Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        x_win = self.attn(x_win, mask=attn_mask)
        x_win = x_win.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_win, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Embed & Patch Merging
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """4×4 non-overlapping patch embedding → (B, H/4*W/4, embed_dim)."""

    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 in_c: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)          # (B, C, H/4, W/4)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, H/4*W/4, C)
        return self.norm(x), H, W


class PatchMerging(nn.Module):
    """Downsample spatial resolution by 2× and double channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = self.reduction(self.norm(x.view(B, -1, 4 * C)))
        return x, H // 2, W // 2


# ---------------------------------------------------------------------------
# Swin Stage
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """Alternating (W-MSA, SW-MSA) blocks + optional PatchMerging."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 7, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 downsample: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def _attn_mask(self, H: int, W: int, device) -> Tensor | None:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_win = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        mask = self._attn_mask(H, W, x.device)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask=mask if blk.shift_size > 0 else None)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


# ---------------------------------------------------------------------------
# SwinTiny
# ---------------------------------------------------------------------------

class SwinTiny(nn.Module):
    """Swin-Tiny with multi-task classification head.

    Architecture:
        PatchEmbed (4×4) → 4 stages (depths 2,2,6,2, dims 96→768) → LayerNorm → GAP → MultiTaskHead
    """

    def __init__(self, head_dropout: float = 0.1) -> None:
        super().__init__()
        embed_dim = 96
        depths    = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        window_size = 7
        dims = [embed_dim * (2 ** i) for i in range(4)]  # [96, 192, 384, 768]

        self.patch_embed = PatchEmbed(img_size=224, patch_size=4, in_c=3, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(0.0)

        self.stages = nn.ModuleList([
            SwinStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, downsample=(i < 3),
            )
            for i in range(4)
        ])

        self.norm = nn.LayerNorm(dims[-1])
        self.head = MultiTaskHead(in_features=dims[-1], dropout=head_dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x, H, W = self.patch_embed(x)   # (B, 56*56, 96)
        x = self.pos_drop(x)

        for stage in self.stages:
            x, H, W = stage(x, H, W)   # final: (B, 49, 768)

        x = self.norm(x).mean(dim=1)    # GAP → (B, 768)
        return self.head(x)
