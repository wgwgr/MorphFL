"""Morphometric evidence module (MEM).

The MEM consumes only morphology scalars, validity flags and quality
metadata. It contains:

- a compilation branch: scalar/quality MLPs and regional evidence encoders
  produce compiled regional concepts, supervised by masked regression onto
  the declared geometric targets; a classifier emits the per-sample
  correction logits (Delta z), which is the one MEM component reading the
  shared base concept ``u``;
- a scalar-only anchor branch: layer normalization and an anchor MLP map the
  validity-masked scalars and the validity vector to class logits.
"""


from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.Layers import MLP, MLPHead, masked_regression


class MorphometricEvidenceModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        scalar_feat_dim: int,
        concept_dim: int,
        head_target_dim: int,
        tail_target_dim: int,
        compiler_input_dim: int = 6,
        compiler_hidden_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        content_conflict_decay: float = 0.30,
        content_conflict_anchor_strength: float = 0.85,
        compiler_conflict_decay: float = 0.75,
        compiler_conflict_anchor_strength: float = 0.85,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.content_conflict_decay = float(min(max(content_conflict_decay, 0.0), 1.0))
        self.content_conflict_anchor_strength = float(
            min(max(content_conflict_anchor_strength, 0.0), 1.0)
        )
        self.compiler_conflict_decay = float(min(max(compiler_conflict_decay, 0.0), 1.0))
        self.compiler_conflict_anchor_strength = float(
            min(max(compiler_conflict_anchor_strength, 0.0), 1.0)
        )

        self.compiler_head = MLP(
            input_dim=int(compiler_input_dim),
            output_dim=concept_dim,
            hidden_dim=max(int(compiler_hidden_dim), concept_dim),
            dropout=float(dropout),
        )
        self.compiler_tail = MLP(
            input_dim=int(compiler_input_dim),
            output_dim=concept_dim,
            hidden_dim=max(int(compiler_hidden_dim), concept_dim),
            dropout=float(dropout),
        )
        self.compiled_classifier = MLPHead(
            input_dim=(3 * concept_dim) + 3,
            hidden_dim=max(int(hidden_dim), concept_dim),
            output_dim=self.num_classes,
            dropout=float(dropout),
        )
        # The anchor reads scalars and validity only, so its sensitivity to
        # imaging perturbations is limited to the scalar re-estimation noise.
        self.anchor_classifier = MLPHead(
            input_dim=int(scalar_feat_dim) + 1,
            hidden_dim=max(int(hidden_dim), concept_dim),
            output_dim=self.num_classes,
            dropout=float(dropout),
            hidden_dims=[max(int(hidden_dim), concept_dim), 128],
        )
        self.scalar_norm = nn.LayerNorm(int(scalar_feat_dim))
        self.head_morph_predictor = MLP(
            input_dim=concept_dim,
            output_dim=int(head_target_dim),
            hidden_dim=max(concept_dim, 32),
            dropout=float(dropout),
        )
        self.tail_morph_predictor = MLP(
            input_dim=concept_dim,
            output_dim=int(tail_target_dim),
            hidden_dim=max(concept_dim, 32),
            dropout=float(dropout),
        )
        self.head_evidence_encoder = MLP(
            input_dim=int(head_target_dim) + 4,
            output_dim=concept_dim,
            hidden_dim=max(int(compiler_hidden_dim), concept_dim),
            dropout=float(dropout),
        )
        self.tail_evidence_encoder = MLP(
            input_dim=int(tail_target_dim) + 4,
            output_dim=concept_dim,
            hidden_dim=max(int(compiler_hidden_dim), concept_dim),
            dropout=float(dropout),
        )

    def expand_valid_mask(self, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.to(device=target.device, dtype=target.dtype).view(target.size(0), -1)
        if valid.size(1) == target.size(1):
            return valid
        if valid.size(1) == 1:
            return valid.expand(-1, target.size(1))
        raise ValueError(
            f"Morph valid mask dim {valid.size(1)} does not match target dim {target.size(1)}."
        )

    def anchor(self, scalar_feats_masked: torch.Tensor, scalar_valid: torch.Tensor) -> torch.Tensor:
        return self.anchor_classifier(
            torch.cat([self.scalar_norm(scalar_feats_masked), scalar_valid], dim=1)
        )

    def evidence_tokens(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        evidence_strength: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        head_target = compiler_bundle["head_morph_target"]
        tail_target = compiler_bundle["tail_morph_target"]
        head_valid = self.expand_valid_mask(head_target, compiler_bundle["head_morph_valid"])
        tail_valid = self.expand_valid_mask(tail_target, compiler_bundle["tail_morph_valid"])

        head_source_confidence = (
            head_valid.mean(dim=1, keepdim=True)
            * compiler_bundle["head_reliability"]
            * evidence_strength
        ).clamp(0.0, 1.0)
        tail_source_confidence = (
            tail_valid.mean(dim=1, keepdim=True)
            * compiler_bundle["tail_reliability"]
            * evidence_strength
        ).clamp(0.0, 1.0)

        head_inputs = torch.cat(
            [
                head_target * head_valid,
                compiler_bundle["head_quality"],
                compiler_bundle["head_reliability"],
                head_source_confidence,
                evidence_strength,
            ],
            dim=1,
        )
        tail_inputs = torch.cat(
            [
                tail_target * tail_valid,
                compiler_bundle["tail_quality"],
                compiler_bundle["tail_reliability"],
                tail_source_confidence,
                evidence_strength,
            ],
            dim=1,
        )
        return {
            "head_inputs": head_inputs,
            "tail_inputs": tail_inputs,
            "head_source_confidence": head_source_confidence,
            "tail_source_confidence": tail_source_confidence,
        }

    def content_gate_prior(
        self,
        source_reliability: torch.Tensor,
        source_confidence: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.15
            + 0.85 * source_reliability.clamp(0.0, 1.0) * source_confidence.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

    def content_gate(
        self,
        source_reliability: torch.Tensor,
        source_confidence: torch.Tensor,
        compiled_conf: torch.Tensor,
        compiled_margin: torch.Tensor,
        conflict_score: torch.Tensor,
    ) -> torch.Tensor:
        prior_gate = self.content_gate_prior(source_reliability, source_confidence)
        source_anchor = (
            source_reliability.clamp(0.0, 1.0) * source_confidence.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        conflict_penalty = (
            self.content_conflict_decay
            * conflict_score.clamp(0.0, 1.0)
            * (1.0 - self.content_conflict_anchor_strength * source_anchor).clamp(0.25, 1.0)
        )
        stability_decay = (
            1.0
            - 0.20 * (1.0 - compiled_conf.clamp(0.0, 1.0))
            - 0.10 * (1.0 - compiled_margin.clamp(0.0, 1.0))
            - conflict_penalty
        ).clamp(0.55, 1.0)
        return (prior_gate * stability_decay).clamp(0.0, 1.0)

    def compiled_statistics(
        self,
        compiled_logits: torch.Tensor,
        core_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compiled_probs = torch.softmax(compiled_logits, dim=1)
        compiled_conf = compiled_probs.max(dim=1, keepdim=True).values
        top2 = torch.topk(compiled_probs, k=min(2, self.num_classes), dim=1).values
        if top2.size(1) == 1:
            compiled_margin = torch.ones_like(compiled_conf)
        else:
            compiled_margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        conflict_score = (
            1.0 - compiled_probs.gather(1, core_probs.argmax(dim=1, keepdim=True)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        return compiled_probs, compiled_conf, compiled_margin, conflict_score

    def compose_concepts(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        metadata_head_concept: torch.Tensor,
        metadata_tail_concept: torch.Tensor,
        head_evidence_concept: torch.Tensor,
        tail_evidence_concept: torch.Tensor,
        head_content_gate: torch.Tensor,
        tail_content_gate: torch.Tensor,
        local_reliability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        compiled_head_concept = metadata_head_concept + head_content_gate * head_evidence_concept
        compiled_tail_concept = metadata_tail_concept + tail_content_gate * tail_evidence_concept
        compiled_head_pred = self.head_morph_predictor(compiled_head_concept)
        compiled_tail_pred = self.tail_morph_predictor(compiled_tail_concept)
        morphology_loss = 0.5 * (
            masked_regression(
                compiled_head_pred,
                compiler_bundle["head_morph_target"],
                compiler_bundle["head_morph_valid"],
            )
            + masked_regression(
                compiled_tail_pred,
                compiler_bundle["tail_morph_target"],
                compiler_bundle["tail_morph_valid"],
            )
        )
        return {
            "compiled_head_concept": compiled_head_concept,
            "compiled_tail_concept": compiled_tail_concept,
            "compiled_local_reliability": local_reliability,
            "compiled_head_content_gate": head_content_gate,
            "compiled_tail_content_gate": tail_content_gate,
            "compiled_head_pred": compiled_head_pred,
            "compiled_tail_pred": compiled_tail_pred,
            "compiled_morphology_loss": morphology_loss,
        }

    def compile_prior(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        evidence_strength: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # The compiler reads raw signals only; derived interactions are left
        # to the MLPs rather than hand-engineered.
        compiler_inputs = torch.cat(
            [
                compiler_bundle["image_quality"],
                compiler_bundle["head_quality"],
                compiler_bundle["tail_quality"],
                compiler_bundle["head_valid"],
                compiler_bundle["tail_sample_valid"],
                evidence_strength,
            ],
            dim=1,
        )
        metadata_head_concept = self.compiler_head(compiler_inputs)
        metadata_tail_concept = self.compiler_tail(compiler_inputs)
        tokens = self.evidence_tokens(compiler_bundle, evidence_strength)
        head_evidence_concept = self.head_evidence_encoder(tokens["head_inputs"])
        tail_evidence_concept = self.tail_evidence_encoder(tokens["tail_inputs"])
        local_reliability = (
            compiler_bundle["local_reliability"] * (0.5 + 0.5 * evidence_strength)
        ).clamp(0.0, 1.0)
        head_gate = self.content_gate_prior(
            compiler_bundle["head_reliability"], tokens["head_source_confidence"]
        )
        tail_gate = self.content_gate_prior(
            compiler_bundle["tail_reliability"], tokens["tail_source_confidence"]
        )
        outputs = self.compose_concepts(
            compiler_bundle=compiler_bundle,
            metadata_head_concept=metadata_head_concept,
            metadata_tail_concept=metadata_tail_concept,
            head_evidence_concept=head_evidence_concept,
            tail_evidence_concept=tail_evidence_concept,
            head_content_gate=head_gate,
            tail_content_gate=tail_gate,
            local_reliability=local_reliability,
        )
        outputs.update(
            {
                "metadata_head_concept": metadata_head_concept,
                "metadata_tail_concept": metadata_tail_concept,
                "head_evidence_concept": head_evidence_concept,
                "tail_evidence_concept": tail_evidence_concept,
                "head_source_confidence": tokens["head_source_confidence"],
                "tail_source_confidence": tokens["tail_source_confidence"],
            }
        )
        return outputs

    def correction_logits(
        self,
        base_concept: torch.Tensor,
        compiled: Dict[str, torch.Tensor],
        compiler_bundle: Dict[str, torch.Tensor],
        evidence_strength: torch.Tensor,
    ) -> torch.Tensor:
        return self.compiled_classifier(
            torch.cat(
                [
                    base_concept,
                    compiled["compiled_head_concept"],
                    compiled["compiled_tail_concept"],
                    compiled["compiled_local_reliability"],
                    compiler_bundle["local_available"] * evidence_strength,
                    evidence_strength,
                ],
                dim=1,
            )
        )

    def compiler_gate(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        local_reliability: torch.Tensor,
        evidence_strength: torch.Tensor,
        compiled_conf: torch.Tensor,
        compiled_margin: torch.Tensor,
        conflict_score: torch.Tensor,
    ) -> torch.Tensor:
        availability_gate = compiler_bundle["local_available"].clamp(0.0, 1.0)
        reliability_gate = (0.10 + 0.90 * local_reliability).clamp(0.0, 1.0)
        evidence_gate = evidence_strength.clamp(0.0, 1.0)
        morph_anchor = (
            local_reliability.clamp(0.0, 1.0) * evidence_strength.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        conflict_penalty = (
            self.compiler_conflict_decay
            * conflict_score.clamp(0.0, 1.0)
            * (1.0 - self.compiler_conflict_anchor_strength * morph_anchor).clamp(0.20, 1.0)
        )
        stability_gate = (
            1.0
            - 0.50 * (1.0 - compiled_conf.clamp(0.0, 1.0))
            - 0.35 * (1.0 - compiled_margin.clamp(0.0, 1.0))
            - conflict_penalty
        ).clamp(0.20, 1.0)
        return availability_gate * reliability_gate * evidence_gate * stability_gate
