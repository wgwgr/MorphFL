"""Unified reliability calibration (URC).

Two lightweight MLP heads score per-sample path correctness:

- the measurement head rho_m reads measurement-side signals only
  (normalized scalars, quality/validity metadata, anchor confidence and
  margin), so no visual feature enters it;
- the visual head rho_v reads the visual core confidence and margin, the
  style-branch magnitude, evidence strength, scalar availability and the
  anchor-core disagreement.

Both heads are trained with BCE against detached realized-correctness
targets. At inference the log-odds difference gates the fusion
(Eq. 4 of the paper): gamma = clip(sigmoid(rho_m - rho_v + b), 0, gamma_max).
"""


import torch
import torch.nn as nn

from model.Layers import MLPHead


class ReliabilityCalibration(nn.Module):
    def __init__(
        self,
        scalar_feat_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.measurement_head = MLPHead(
            input_dim=int(scalar_feat_dim) + 5,
            hidden_dim=max(int(hidden_dim), 16),
            output_dim=1,
            dropout=float(dropout),
        )
        self.visual_head = MLPHead(
            input_dim=6,
            hidden_dim=max(int(hidden_dim), 16),
            output_dim=1,
            dropout=float(dropout),
        )
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def measurement_logits(self, measurement_input: torch.Tensor) -> torch.Tensor:
        return self.measurement_head(measurement_input)

    def visual_logits(self, visual_input: torch.Tensor) -> torch.Tensor:
        return self.visual_head(visual_input)

    def gate(self, measurement_logits: torch.Tensor, visual_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(measurement_logits - visual_logits + self.gate_bias)

    @staticmethod
    def warmup_alpha(current_round: int, warmup_rounds: int) -> float:
        return min(max((int(current_round) - 1) / float(max(int(warmup_rounds), 1)), 0.0), 1.0)
