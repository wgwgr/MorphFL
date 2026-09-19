"""Visual semantic projection (VSP).

The VSP is the only image-derived function class that crosses the federated
boundary. It maps the private CLS embedding through:

1. a shared rank-16 residual adapter producing ``h``;
2. a shared projector into the 64-d shared base concept ``u``;
3. a shared context encoder fusing ``h``, ``u`` and a client-side
   measurement summary into the shared context ``g``;
4. shared visual decision heads on ``[h, u, g]``.

A bounded private style branch (rank-16 residual adapter, norm clamp and a
tanh-bounded calibrator) absorbs residual site bias and stays client-local.
"""


from typing import Dict, Optional

import torch
import torch.nn as nn

from model.Layers import LowRankAdapter, MLP, MLPHead


class VisualSemanticProjection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hierarchical: bool,
        concept_dim: int = 64,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        share_global_context: bool = True,
        style_residual_scale: float = 0.12,
        style_logit_scale: float = 0.05,
        style_delta_max_norm: float = 0.60,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.concept_dim = int(concept_dim)
        self.share_global_context = bool(share_global_context)
        self.style_residual_scale = max(float(style_residual_scale), 0.0)
        self.style_logit_scale = max(float(style_logit_scale), 0.0)
        self.style_delta_max_norm = max(float(style_delta_max_norm), 0.0)

        self.morph_adapter = LowRankAdapter(
            dim=self.embed_dim,
            rank=int(adapter_rank),
            alpha=float(adapter_alpha),
            dropout=float(dropout),
        )
        self.style_adapter = LowRankAdapter(
            dim=self.embed_dim,
            rank=int(adapter_rank),
            alpha=float(adapter_alpha),
            dropout=float(dropout),
        )
        self.concept_projector = MLP(
            input_dim=self.embed_dim,
            output_dim=self.concept_dim,
            hidden_dim=max(self.embed_dim, self.concept_dim),
            dropout=float(dropout),
        )
        self.context_encoder = MLP(
            input_dim=self.embed_dim + (2 * self.concept_dim) + 2,
            output_dim=self.concept_dim,
            hidden_dim=max(self.embed_dim, self.concept_dim),
            dropout=float(dropout),
        )
        self.local_summary_encoder = MLP(
            input_dim=5,
            output_dim=self.concept_dim,
            hidden_dim=max(self.concept_dim // 2, 16),
            dropout=float(dropout),
        )
        self.style_calibrator = MLPHead(
            input_dim=self.embed_dim + self.concept_dim,
            hidden_dim=int(hidden_dim),
            output_dim=int(num_classes),
            dropout=float(dropout),
        )

        decision_dim = self.embed_dim + (2 * self.concept_dim)
        if hierarchical:
            self.ns_gate = MLPHead(decision_dim, int(hidden_dim), 1, dropout=float(dropout))
            self.abn_cls = MLPHead(decision_dim, int(hidden_dim), 2, dropout=float(dropout))
            self.base_classifier = None
        else:
            self.ns_gate = None
            self.abn_cls = None
            self.base_classifier = MLPHead(
                decision_dim, int(hidden_dim), int(num_classes), dropout=float(dropout)
            )

    def limit_style_delta(self, style_delta: torch.Tensor) -> torch.Tensor:
        if self.style_delta_max_norm <= 0:
            return style_delta
        norm = style_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(float(self.style_delta_max_norm) / norm, max=1.0)
        return style_delta * scale

    def adapt(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        morph_feat, _ = self.morph_adapter(feat)
        _, style_delta = self.style_adapter(feat)
        effective_style_delta = self.style_residual_scale * self.limit_style_delta(style_delta)
        if style_delta.dim() == 2:
            style_delta_norm = style_delta.norm(dim=-1)
        else:
            style_delta_norm = style_delta.norm(dim=-1).mean(dim=1)
        return {
            "morph_feat": morph_feat,
            "style_delta": style_delta,
            "effective_style_delta": effective_style_delta,
            "style_delta_norm": style_delta_norm,
        }

    def base_concept(self, morph_feat: torch.Tensor) -> torch.Tensor:
        return self.concept_projector(morph_feat)

    def build_context(
        self,
        morph_feat: torch.Tensor,
        base_concept: torch.Tensor,
        region: Dict[str, torch.Tensor],
        evidence_quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.share_global_context:
            return morph_feat.new_zeros((morph_feat.size(0), self.concept_dim))
        if evidence_quality is None:
            evidence_quality = morph_feat.new_ones((morph_feat.size(0), 1))
        cheap_summary = torch.cat(
            [
                region["image_quality"],
                region["head_quality"],
                region["tail_quality"],
                region["head_valid"],
                region["tail_sample_valid"],
            ],
            dim=1,
        )
        cheap_local_concept = self.local_summary_encoder(cheap_summary)
        conditioned_head_valid = region["head_valid"] * evidence_quality
        conditioned_tail_valid = region["tail_sample_valid"] * evidence_quality
        local_weight = conditioned_head_valid + conditioned_tail_valid
        expensive_num = (
            region["head_concept"] * conditioned_head_valid
            + region["tail_concept"] * conditioned_tail_valid
        )
        expensive_local_concept = torch.where(
            local_weight > 0.0,
            expensive_num / local_weight.clamp_min(1e-6),
            torch.zeros_like(base_concept),
        )
        replace_weight = region["local_available"].clamp(0.0, 1.0) * evidence_quality
        # Every sample keeps the dense cheap summary; where compiled regional
        # concepts are available they replace it in proportion to availability.
        local_concept = cheap_local_concept + replace_weight * (
            expensive_local_concept - cheap_local_concept
        )
        context_input = torch.cat(
            [
                morph_feat,
                base_concept,
                local_concept,
                region["local_available"] * evidence_quality,
                region["local_reliability"] * evidence_quality,
            ],
            dim=1,
        )
        return self.context_encoder(context_input)

    def style_calibration(
        self,
        effective_style_delta: torch.Tensor,
        global_context: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.style_calibrator(torch.cat([effective_style_delta, global_context], dim=1))
        logits = float(self.style_logit_scale) * torch.tanh(logits)
        if reliability is not None:
            logits = logits * (0.2 + 0.8 * reliability).clamp(0.0, 1.0)
        return logits

    def core_outputs(
        self,
        morph_feat: torch.Tensor,
        base_concept: torch.Tensor,
        global_context: torch.Tensor,
        style_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        decision_feat = torch.cat([morph_feat, base_concept, global_context], dim=1)
        if self.ns_gate is not None:
            ns_logit = self.ns_gate(decision_feat)
            abn_logits = self.abn_cls(decision_feat)
            p_ns = torch.sigmoid(ns_logit).squeeze(-1)
            p_abn = torch.softmax(abn_logits, dim=1)
            probs = morph_feat.new_zeros((morph_feat.size(0), 3))
            probs[:, 2] = p_ns
            probs[:, 0] = (1.0 - p_ns) * p_abn[:, 0]
            probs[:, 1] = (1.0 - p_ns) * p_abn[:, 1]
            main_logits = torch.log(probs.clamp_min(1e-12))
        else:
            main_logits = self.base_classifier(decision_feat)
        logits = main_logits if style_logits is None else main_logits + style_logits
        return {"logits": logits, "probs": torch.softmax(logits, dim=1)}
