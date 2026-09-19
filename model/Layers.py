"""Shared neural building blocks.

These layers carry no protocol semantics on their own; they are reused by the
VSP, MEM and URC modules declared in :mod:`BoundaryScope`.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_regression(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean smooth-L1 loss over valid (mask value 1) elements and samples."""
    mask = valid.to(device=pred.device, dtype=pred.dtype)
    if mask.size(1) == 1 and pred.size(1) != 1:
        mask = mask.expand(-1, pred.size(1))
    elif mask.size(1) != pred.size(1):
        raise ValueError(f"Regression valid mask dim {mask.size(1)} does not match prediction dim {pred.size(1)}.")
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    per_sample = (per_elem * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    sample_mask = (mask.sum(dim=1) > 0.0).to(dtype=pred.dtype)
    if int(sample_mask.sum()) == 0:
        return pred.new_zeros(())
    return (per_sample * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


class LowRankAdapter(nn.Module):
    """Low-rank residual adapter with optional norm clamp on the delta."""

    def __init__(self, dim: int, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / max(int(rank), 1)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.down(self.norm(x))
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        delta = self.up(hidden) * self.scale
        return x + delta, delta


class MLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear projection."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """LayerNorm followed by a configurable-depth MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dims = list(hidden_dims) if hidden_dims is not None else [hidden_dim]
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for hidden in dims:
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
