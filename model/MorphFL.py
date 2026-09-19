"""MorphFL client model: assembly of the morphology-basis protocol.

The model combines the private visual frontend with the three shared
function classes (VSP, MEM, URC). Images, the encoder, the LoRA adapters and
the style branch stay client-local; the shared surface is exactly the VSP,
MEM and URC parameters declared in :mod:`morphfl.federated.MBP`.
"""


import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.DomainShift import canonicalize_domain_profile, domain_profile_cache_key
from utils.Morphology import (
    HeadRegionExtractor,
    MorphologyExtractor,
    SmidsAnatomyCache,
    TailDescriptor,
    TailRegionExtractor,
    fit_mask_ellipse,
)
from model.MEM import MorphometricEvidenceModule
from model.URC import ReliabilityCalibration
from model.VisualFrontend import PrivateVisualFrontend
from model.VSP import VisualSemanticProjection


_SHARED_EXTRACTOR_BUNDLES: Dict[Tuple[str, int, str], Dict[str, object]] = {}


def _get_shared_extractor_bundle(
    data_dir: str,
    image_size: int,
    domain_shift_profile=None,
) -> Dict[str, object]:
    """Share one morphology extraction pipeline per (dataset, size, profile)."""
    profile = canonicalize_domain_profile(domain_shift_profile)
    key = (
        os.path.abspath(data_dir),
        int(image_size),
        domain_profile_cache_key(profile),
    )
    bundle = _SHARED_EXTRACTOR_BUNDLES.get(key)
    if bundle is not None:
        return bundle

    extractor = MorphologyExtractor(data_dir=key[0], image_size=key[1], domain_shift_profile=profile)
    bundle = {
        "head_extractor": HeadRegionExtractor(data_dir=key[0], image_size=key[1], extractor=extractor),
        "tail_extractor": TailRegionExtractor(data_dir=key[0], image_size=key[1], extractor=extractor),
        "tail_descriptor": TailDescriptor(data_dir=key[0], image_size=key[1], extractor=extractor),
        "smids_anatomy_cache": SmidsAnatomyCache(data_dir=key[0], domain_shift_profile=profile),
    }
    _SHARED_EXTRACTOR_BUNDLES[key] = bundle
    return bundle


class MorphFLModel(nn.Module):
    SMIDS_HEAD_TARGET_NAMES = (
        "head_ratio_z",
        "head_ratio_band_dev",
        "head_eccentricity",
        "head_ellipse_coverage",
        "head_solidity",
        "head_circularity",
    )
    SMIDS_TAIL_TARGET_NAMES = (
        "tail_off_axis_ratio",
        "tail_near_root_ratio",
        "tail_component_ratio",
    )

    def __init__(
        self,
        data_dir: str,
        pretrained_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        train_norm_layers: bool = False,
        domain_shift_profile=None,
        attn_implementation: str = "sdpa",
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        use_hierarchical_head: bool = True,
        scalar_feat_dim: int = 2,
        concept_dim: int = 64,
        morph_adapter_rank: int = 16,
        morph_adapter_alpha: float = 16.0,
        adapter_dropout: float = 0.1,
        hidden_dim: int = 256,
        share_global_context: bool = True,
        style_residual_scale: float = 0.12,
        style_logit_scale: float = 0.05,
        style_delta_max_norm: float = 0.60,
        compiler_input_dim: int = 6,
        compiler_hidden_dim: int = 128,
        content_conflict_decay: float = 0.30,
        content_conflict_anchor_strength: float = 0.85,
        compiler_conflict_decay: float = 0.75,
        compiler_conflict_anchor_strength: float = 0.85,
        anchor_floor: float = 0.0,
        anchor_scale: float = 0.7,
        anchor_conflict_boost: float = 0.80,
        anchor_max: float = 0.95,
        anchor_takeover_mid: float = 0.45,
        anchor_takeover_width: float = 0.10,
        rho_enabled: bool = False,
        rho_warmup_rounds: int = 4,
        rho_hidden_dim: int = 32,
    ):
        """Assemble the private frontend and the shared VSP/MEM/(URC) modules."""
        super().__init__()
        self.frontend = PrivateVisualFrontend(
            pretrained_model_name=pretrained_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            train_norm_layers=train_norm_layers,
            attn_implementation=str(attn_implementation),
        )
        self.backbone_type = self.frontend.backbone_type
        self.embed_dim = self.frontend.embed_dim
        self.num_classes = int(num_classes)
        self.use_hierarchical_head = bool(use_hierarchical_head)
        self.scalar_feat_dim = int(scalar_feat_dim)
        self.concept_dim = int(concept_dim)
        self.current_round = 1
        self.total_rounds = 1
        self.domain_shift_profile = canonicalize_domain_profile(domain_shift_profile)
        self.anchor_floor = float(min(max(anchor_floor, 0.0), 1.0))
        self.anchor_scale = float(min(max(anchor_scale, 0.0), 1.0))
        self.anchor_conflict_boost = float(min(max(anchor_conflict_boost, 0.0), 1.0))
        self.anchor_max = float(min(max(anchor_max, 0.0), 1.0))
        self.anchor_takeover_mid = float(anchor_takeover_mid)
        self.anchor_takeover_width = float(max(anchor_takeover_width, 1e-3))

        self.smids_head_target_names = list(self.SMIDS_HEAD_TARGET_NAMES)
        self.smids_tail_target_indices = [
            int(TailDescriptor.FEATURE_NAMES.index(name))
            for name in self.SMIDS_TAIL_TARGET_NAMES
        ]

        self.vsp = VisualSemanticProjection(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            hierarchical=self.use_hierarchical_head,
            concept_dim=self.concept_dim,
            adapter_rank=int(morph_adapter_rank),
            adapter_alpha=float(morph_adapter_alpha),
            dropout=float(adapter_dropout),
            hidden_dim=int(hidden_dim),
            share_global_context=bool(share_global_context),
            style_residual_scale=float(style_residual_scale),
            style_logit_scale=float(style_logit_scale),
            style_delta_max_norm=float(style_delta_max_norm),
        )

        if self.use_hierarchical_head:
            head_target_dim = len(self.smids_head_target_names)
            tail_target_dim = len(self.smids_tail_target_indices)
            self.nucleus_scalar_feat_dim = 0
            self.cytoplasm_scalar_feat_dim = 0
        else:
            if self.scalar_feat_dim % 2 != 0:
                raise ValueError(
                    "SIPaKMeD configuration expects an even scalar_feat_dim so nucleus/"
                    "cytoplasm targets split evenly."
                )
            head_target_dim = self.scalar_feat_dim // 2
            tail_target_dim = self.scalar_feat_dim // 2
            self.nucleus_scalar_feat_dim = head_target_dim
            self.cytoplasm_scalar_feat_dim = tail_target_dim

        self.mem = MorphometricEvidenceModule(
            num_classes=self.num_classes,
            scalar_feat_dim=self.scalar_feat_dim,
            concept_dim=self.concept_dim,
            head_target_dim=head_target_dim,
            tail_target_dim=tail_target_dim,
            compiler_input_dim=int(compiler_input_dim),
            compiler_hidden_dim=int(compiler_hidden_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(adapter_dropout),
            content_conflict_decay=float(content_conflict_decay),
            content_conflict_anchor_strength=float(content_conflict_anchor_strength),
            compiler_conflict_decay=float(compiler_conflict_decay),
            compiler_conflict_anchor_strength=float(compiler_conflict_anchor_strength),
        )

        self.rho_enabled = bool(rho_enabled)
        self.rho_warmup_rounds = max(int(rho_warmup_rounds), 1)
        if self.rho_enabled:
            self.urc = ReliabilityCalibration(
                scalar_feat_dim=self.scalar_feat_dim,
                hidden_dim=int(rho_hidden_dim),
                dropout=float(adapter_dropout),
            )

        if self.use_hierarchical_head:
            extractor_bundle = _get_shared_extractor_bundle(
                data_dir=data_dir,
                image_size=224,
                domain_shift_profile=self.domain_shift_profile,
            )
            self.head_extractor = extractor_bundle["head_extractor"]
            self.tail_extractor = extractor_bundle["tail_extractor"]
            self.tail_descriptor = extractor_bundle["tail_descriptor"]
            self.smids_anatomy_cache = extractor_bundle.get("smids_anatomy_cache")
        else:
            self.head_extractor = None
            self.tail_extractor = None
            self.tail_descriptor = None
            self.smids_anatomy_cache = None

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  MorphFLModel | {self.backbone_type.upper()} backbone ({pretrained_model_name}) | "
            f"trainable: {trainable:,}"
        )

    def set_round_context(self, current_round: int, total_rounds: int) -> None:
        """Record the current federated round for warm-up-aware gating."""
        self.current_round = max(int(current_round), 1)
        self.total_rounds = max(int(total_rounds), 1)

    def _resolve_quality_signals(
        self,
        image_quality: torch.Tensor,
        head_quality: torch.Tensor,
        tail_quality: torch.Tensor,
        head_valid: torch.Tensor,
        tail_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Combine validity flags and quality scores into reliability signals."""
        image_quality = image_quality.clamp(0.0, 1.0)
        head_quality = head_quality.clamp(0.0, 1.0)
        tail_quality = tail_quality.clamp(0.0, 1.0)
        tail_sample_valid = torch.max(tail_valid, dim=1, keepdim=True).values
        local_available = torch.maximum(head_valid, tail_sample_valid)
        head_reliability = head_valid * head_quality
        tail_reliability = tail_sample_valid * tail_quality
        local_reliability = torch.maximum(head_reliability, tail_reliability)
        return {
            "tail_sample_valid": tail_sample_valid,
            "local_available": local_available,
            "head_reliability": head_reliability,
            "tail_reliability": tail_reliability,
            "local_reliability": local_reliability,
        }

    def _split_sipak_region_targets(self, scalar_feats_masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the released SIPaKMeD scalars into nucleus/cytoplasm halves."""
        split = int(self.nucleus_scalar_feat_dim)
        return scalar_feats_masked[:, :split], scalar_feats_masked[:, split:]

    def _extract_smids_head_targets(
        self,
        scalar_feats_masked: torch.Tensor,
        scalar_valid: torch.Tensor,
        head_masks_np: Optional[np.ndarray],
        head_valid: torch.Tensor,
        head_geometry_np: Optional[np.ndarray] = None,
        head_geometry_valid_np: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build head regression targets: released scalars plus geometry.

        The geometry part (eccentricity, ellipse coverage, solidity,
        circularity) is taken from the anatomy cache when available or
        recomputed from the head masks; validity is masked by head_valid.
        """
        if head_geometry_np is not None and head_geometry_valid_np is not None:
            geom_targets_t = torch.from_numpy(np.asarray(head_geometry_np, dtype=np.float32)).to(
                device=scalar_feats_masked.device,
                dtype=scalar_feats_masked.dtype,
            )
            geom_valid_t = torch.from_numpy(np.asarray(head_geometry_valid_np, dtype=np.float32)).to(
                device=scalar_feats_masked.device,
                dtype=scalar_feats_masked.dtype,
            )
        else:
            geom_targets, geom_valid = [], []
            for mask in head_masks_np:
                mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
                area = float(mask_u8.sum())
                if area < 16:
                    geom_targets.append(np.zeros(4, dtype=np.float32))
                    geom_valid.append(np.zeros(4, dtype=np.float32))
                    continue
                ellipse = fit_mask_ellipse(mask_u8)
                if ellipse is None:
                    geom_targets.append(np.zeros(4, dtype=np.float32))
                    geom_valid.append(np.zeros(4, dtype=np.float32))
                    continue
                (_, _), (v1, v2), _ = ellipse
                major = float(max(v1, v2))
                minor = float(max(min(v1, v2), 1e-6))
                ellipse_area = float(np.pi * (major / 2.0) * (minor / 2.0))
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-8))))
                ellipse_coverage = float(np.clip(1.0 - abs(area / max(ellipse_area, 1e-6) - 1.0), 0.0, 1.0))
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    geom_targets.append(np.zeros(4, dtype=np.float32))
                    geom_valid.append(np.zeros(4, dtype=np.float32))
                    continue
                contour = max(contours, key=cv2.contourArea)
                perimeter = float(cv2.arcLength(contour, True))
                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                solidity = float(np.clip(area / max(hull_area, 1e-6), 0.0, 1.0))
                circularity = float(
                    np.clip((4.0 * np.pi * area) / max(perimeter * perimeter, 1e-6), 0.0, 1.0)
                )
                geom_targets.append(
                    np.array([eccentricity, ellipse_coverage, solidity, circularity], dtype=np.float32)
                )
                geom_valid.append(np.ones(4, dtype=np.float32))

            geom_targets_t = torch.from_numpy(np.stack(geom_targets, axis=0)).to(
                device=scalar_feats_masked.device,
                dtype=scalar_feats_masked.dtype,
            )
            geom_valid_t = torch.from_numpy(np.stack(geom_valid, axis=0)).to(
                device=scalar_feats_masked.device,
                dtype=scalar_feats_masked.dtype,
            )
        geom_valid_t = geom_valid_t * head_valid.expand(-1, geom_targets_t.size(1))
        scalar_valid_expand = scalar_valid.to(
            device=scalar_feats_masked.device, dtype=scalar_feats_masked.dtype
        ).expand(-1, scalar_feats_masked.size(1))
        targets = torch.cat([scalar_feats_masked, geom_targets_t], dim=1)
        valid = torch.cat([scalar_valid_expand, geom_valid_t], dim=1)
        return targets, valid

    def _extract_smids_tail_targets(self, tail_descriptor: np.ndarray, device, dtype) -> torch.Tensor:
        """Select the declared tail descriptor columns as weak targets."""
        target = tail_descriptor[:, self.smids_tail_target_indices]
        return torch.from_numpy(target).to(device=device, dtype=dtype)

    def _build_compiler_bundle(
        self,
        images: torch.Tensor,
        scalar_feats_masked: torch.Tensor,
        scalar_valid: torch.Tensor,
        relpaths: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Collect measurement-side evidence for the MEM compiler.

        For SMIDS this reads the per-image anatomy cache when available
        (falling back to recomputing head/tail evidence) and derives the
        per-sample quality/reliability signals and regression targets;
        SIPaKMeD reuses the released table scalars split per region.
        """
        if self.use_hierarchical_head:
            cache_items = None
            cache_indices = None
            if self.smids_anatomy_cache is not None and getattr(self.smids_anatomy_cache, "available", False):
                try:
                    cache_indices = self.smids_anatomy_cache.batch_lookup_indices(relpaths)
                    cache = self.smids_anatomy_cache.cache
                    cache_items = {
                        "head_valid": np.asarray(cache["head_valid"][cache_indices], dtype=np.float32),
                        "head_quality": np.asarray(cache["head_quality"][cache_indices], dtype=np.float32),
                        "image_quality": np.asarray(cache["image_quality"][cache_indices], dtype=np.float32),
                        "tail_valid": np.asarray(cache["tail_valid"][cache_indices], dtype=np.float32),
                        "tail_quality": np.asarray(cache["tail_quality"][cache_indices], dtype=np.float32),
                    }
                except (FileNotFoundError, KeyError):
                    cache_items = None
            if cache_items is not None:
                head_valid = torch.from_numpy(cache_items["head_valid"]).to(
                    device=images.device, dtype=images.dtype
                )
                head_quality = torch.from_numpy(cache_items["head_quality"]).to(
                    device=images.device, dtype=images.dtype
                )
                image_quality = torch.from_numpy(cache_items["image_quality"]).to(
                    device=images.device, dtype=images.dtype
                )
                tail_valid = torch.from_numpy(cache_items["tail_valid"]).to(
                    device=images.device, dtype=images.dtype
                )
                tail_quality = torch.from_numpy(cache_items["tail_quality"]).to(
                    device=images.device, dtype=images.dtype
                )
                head_targets, head_target_valid = self._extract_smids_head_targets(
                    scalar_feats_masked=scalar_feats_masked,
                    scalar_valid=scalar_valid,
                    head_masks_np=None,
                    head_valid=head_valid,
                    head_geometry_np=np.asarray(
                        self.smids_anatomy_cache.cache["head_geometry"][cache_indices], dtype=np.float32
                    ),
                    head_geometry_valid_np=np.asarray(
                        self.smids_anatomy_cache.cache["head_geometry_valid"][cache_indices],
                        dtype=np.float32,
                    ),
                )
                weak_np = np.asarray(
                    self.smids_anatomy_cache.cache["tail_struct"][cache_indices], dtype=np.float32
                )
                tail_target = torch.from_numpy(weak_np).to(device=images.device, dtype=images.dtype)
            else:
                head_items = [self.head_extractor.extract(rel) for rel in relpaths]
                head_masks_np = np.stack([item["head_mask"] for item in head_items], axis=0).astype(np.float32)
                head_valid = torch.from_numpy(
                    np.stack([item["valid"] for item in head_items], axis=0).astype(np.float32)
                ).to(device=images.device, dtype=images.dtype)
                head_quality = torch.from_numpy(
                    np.stack([item["head_quality"] for item in head_items], axis=0).astype(np.float32)
                ).to(device=images.device, dtype=images.dtype)
                image_quality = torch.from_numpy(
                    np.stack([item["image_quality"] for item in head_items], axis=0).astype(np.float32)
                ).to(device=images.device, dtype=images.dtype)
                head_targets, head_target_valid = self._extract_smids_head_targets(
                    scalar_feats_masked=scalar_feats_masked,
                    scalar_valid=scalar_valid,
                    head_masks_np=head_masks_np,
                    head_valid=head_valid,
                )
                tail_items = [self.tail_extractor.extract(rel) for rel in relpaths]
                tail_valid = torch.from_numpy(
                    np.stack([item["valid"] for item in tail_items], axis=0).astype(np.float32)
                ).to(device=images.device, dtype=images.dtype)
                tail_quality = torch.from_numpy(
                    np.stack([item["tail_quality"] for item in tail_items], axis=0).astype(np.float32)
                ).to(device=images.device, dtype=images.dtype)
                weak_np = np.stack(
                    [self.tail_descriptor.extract(rel) for rel in relpaths], axis=0
                ).astype(np.float32)
                tail_target = self._extract_smids_tail_targets(weak_np, device=images.device, dtype=images.dtype)
            quality = self._resolve_quality_signals(
                image_quality=image_quality,
                head_quality=head_quality,
                tail_quality=tail_quality,
                head_valid=head_valid,
                tail_valid=tail_valid,
            )
            return {
                "image_quality": image_quality,
                "head_quality": head_quality,
                "tail_quality": tail_quality,
                "head_valid": head_valid,
                "tail_valid": tail_valid,
                "tail_sample_valid": quality["tail_sample_valid"],
                "local_available": quality["local_available"],
                "head_reliability": quality["head_reliability"],
                "tail_reliability": quality["tail_reliability"],
                "local_reliability": quality["local_reliability"],
                "head_morph_target": head_targets,
                "tail_morph_target": tail_target,
                "head_morph_valid": head_target_valid,
                "tail_morph_valid": quality["tail_sample_valid"],
            }

        nucleus_target, cytoplasm_target = self._split_sipak_region_targets(scalar_feats_masked)
        head_valid = scalar_valid
        tail_valid = torch.cat([scalar_valid, scalar_valid], dim=1)
        image_quality = images.new_ones((images.size(0), 1))
        head_quality = scalar_valid
        tail_quality = scalar_valid
        quality = self._resolve_quality_signals(
            image_quality=image_quality,
            head_quality=head_quality,
            tail_quality=tail_quality,
            head_valid=head_valid,
            tail_valid=tail_valid,
        )
        return {
            "image_quality": image_quality,
            "head_quality": head_quality,
            "tail_quality": tail_quality,
            "head_valid": head_valid,
            "tail_valid": tail_valid,
            "tail_sample_valid": quality["tail_sample_valid"],
            "local_available": quality["local_available"],
            "head_reliability": quality["head_reliability"],
            "tail_reliability": quality["tail_reliability"],
            "local_reliability": quality["local_reliability"],
            "head_morph_target": nucleus_target,
            "tail_morph_target": cytoplasm_target,
            "head_morph_valid": scalar_valid.expand_as(nucleus_target),
            "tail_morph_valid": scalar_valid.expand_as(cytoplasm_target),
        }

    def _build_core_path(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        compiled: Dict[str, torch.Tensor],
        branch: Dict[str, torch.Tensor],
        base_concept: torch.Tensor,
        evidence_strength: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Run the VSP context path and the MEM correction for one pass."""
        region = {
            "image_quality": compiler_bundle["image_quality"],
            "head_quality": compiler_bundle["head_quality"],
            "tail_quality": compiler_bundle["tail_quality"],
            "head_valid": compiler_bundle["head_valid"],
            "tail_sample_valid": compiler_bundle["tail_sample_valid"],
            "local_available": compiler_bundle["local_available"],
            "head_reliability": compiler_bundle["head_reliability"],
            "tail_reliability": compiler_bundle["tail_reliability"],
            "local_reliability": compiled["compiled_local_reliability"],
            "head_concept": compiled["compiled_head_concept"],
            "tail_concept": compiled["compiled_tail_concept"],
        }
        global_context = self.vsp.build_context(
            branch["morph_feat"],
            base_concept,
            region,
            evidence_quality=evidence_strength,
        )
        style_logits = self.vsp.style_calibration(
            branch["effective_style_delta"],
            global_context,
            reliability=compiled["compiled_local_reliability"],
        )
        core_outputs = self.vsp.core_outputs(
            branch["morph_feat"],
            base_concept=base_concept,
            global_context=global_context,
            style_logits=style_logits,
        )
        correction_logits = self.mem.correction_logits(
            base_concept, compiled, compiler_bundle, evidence_strength
        )
        return global_context, style_logits, core_outputs, correction_logits

    def _run_compile_pass(
        self,
        compiler_bundle: Dict[str, torch.Tensor],
        compiled: Dict[str, torch.Tensor],
        branch: Dict[str, torch.Tensor],
        base_concept: torch.Tensor,
        evidence_strength: torch.Tensor,
        gate_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Tuple]:
        """One MEM compilation pass over the gate-prior-weighted logits."""
        global_context, _, core_outputs, correction_logits = self._build_core_path(
            compiler_bundle, compiled, branch, base_concept, evidence_strength
        )
        core_logits = core_outputs["logits"]
        core_probs = core_outputs["probs"]
        compiled_logits = core_logits + gate_prior * correction_logits
        statistics = self.mem.compiled_statistics(compiled_logits, core_probs)
        return global_context, core_outputs, correction_logits, statistics

    def _heuristic_anchor_gate(
        self,
        scalar_valid: torch.Tensor,
        anchor_probs: torch.Tensor,
        core_probs: torch.Tensor,
        source_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the conflict-driven heuristic anchor takeover gate."""
        anchor_core_conflict = (
            1.0 - anchor_probs.gather(1, core_probs.argmax(dim=1, keepdim=True)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        morph_conflict = (anchor_core_conflict * source_confidence).clamp(0.0, 1.0)
        takeover = torch.sigmoid(
            (morph_conflict - self.anchor_takeover_mid) / self.anchor_takeover_width
        )
        base_gate = (
            self.anchor_floor
            + self.anchor_scale
            * scalar_valid.clamp(0.0, 1.0)
            * (1.0 + self.anchor_conflict_boost * anchor_core_conflict * source_confidence)
        ).clamp(0.0, self.anchor_max)
        gate = (base_gate + (self.anchor_max - base_gate) * takeover).clamp(0.0, self.anchor_max)
        return gate, anchor_core_conflict, takeover

    def _fuse_anchor(
        self,
        anchor_logits: torch.Tensor,
        scalar_feats_masked: torch.Tensor,
        scalar_valid: torch.Tensor,
        compiler_bundle: Dict[str, torch.Tensor],
        compiled: Dict[str, torch.Tensor],
        branch: Dict[str, torch.Tensor],
        evidence_strength: torch.Tensor,
        core_probs: torch.Tensor,
        core_logits: torch.Tensor,
        correction_gate: torch.Tensor,
        correction_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse core, correction and anchor logits (optionally via URC).

        With ``rho_enabled`` the learned reliability heads blend into the
        heuristic anchor gate over the configured warm-up rounds; the fused
        logits and their decision statistics are returned.
        """
        anchor_probs = torch.softmax(anchor_logits, dim=1)
        source_confidence = 0.5 * (
            compiled["head_source_confidence"] + compiled["tail_source_confidence"]
        )
        anchor_gate, anchor_core_conflict, takeover = self._heuristic_anchor_gate(
            scalar_valid, anchor_probs, core_probs, source_confidence
        )

        rho_outputs: Dict[str, torch.Tensor] = {}
        if self.rho_enabled:
            anchor_top2 = torch.topk(anchor_probs.detach(), k=min(2, self.num_classes), dim=1).values
            if anchor_top2.size(1) == 1:
                anchor_conf = anchor_probs.detach().max(dim=1, keepdim=True).values
                anchor_margin = torch.ones_like(anchor_conf)
            else:
                anchor_conf = anchor_top2[:, :1]
                anchor_margin = anchor_top2[:, :1] - anchor_top2[:, 1:2]
            measurement_input = torch.cat(
                [
                    self.mem.scalar_norm(scalar_feats_masked),
                    compiler_bundle["head_quality"].detach(),
                    compiler_bundle["tail_quality"].detach(),
                    compiler_bundle["head_valid"].detach(),
                    anchor_conf,
                    anchor_margin,
                ],
                dim=1,
            )
            rho_m_logits = self.urc.measurement_logits(measurement_input)
            rho_m = torch.sigmoid(rho_m_logits)

            core_top2 = torch.topk(core_probs.detach(), k=min(2, self.num_classes), dim=1).values
            if core_top2.size(1) == 1:
                core_conf = core_probs.detach().max(dim=1, keepdim=True).values
                core_margin = torch.ones_like(core_conf)
            else:
                core_conf = core_top2[:, :1]
                core_margin = core_top2[:, :1] - core_top2[:, 1:2]
            visual_input = torch.cat(
                [
                    core_conf,
                    core_margin,
                    branch["style_delta_norm"].detach().reshape(-1, 1),
                    evidence_strength.detach(),
                    compiler_bundle["local_available"].detach(),
                    anchor_core_conflict.detach(),
                ],
                dim=1,
            )
            rho_v_logits = self.urc.visual_logits(visual_input)
            rho_v = torch.sigmoid(rho_v_logits)
            learned_gate = self.urc.gate(rho_m_logits, rho_v_logits).clamp(0.0, self.anchor_max)
            # Linear warm-up: blend the heuristic gate into the learned gate.
            alpha = ReliabilityCalibration.warmup_alpha(self.current_round, self.rho_warmup_rounds)
            anchor_gate = ((1.0 - alpha) * anchor_gate + alpha * learned_gate).clamp(
                0.0, self.anchor_max
            )
            rho_outputs = {
                "rho_m": rho_m,
                "rho_v": rho_v,
                "rho_m_logits": rho_m_logits,
                "rho_v_logits": rho_v_logits,
                "rho_gate": learned_gate,
                "rho_warmup_alpha": anchor_logits.new_tensor(float(alpha)),
            }

        fused_logits = (
            core_logits
            + correction_gate * correction_logits
            + anchor_gate * (anchor_logits - core_logits)
        )
        fused_probs, fused_conf, fused_margin, fused_conflict = self.mem.compiled_statistics(
            fused_logits, core_probs
        )
        result = {
            "anchor_logits": anchor_logits,
            "anchor_probs": anchor_probs,
            "anchor_core_conflict": anchor_core_conflict,
            "anchor_takeover": takeover,
            "anchor_gate": anchor_gate,
            "fused_logits": fused_logits,
            "fused_probs": fused_probs,
            "fused_conf": fused_conf,
            "fused_margin": fused_margin,
            "fused_conflict_score": fused_conflict,
        }
        result.update(rho_outputs)
        return result

    def forward(
        self,
        images: torch.Tensor,
        scalar_feats: torch.Tensor,
        relpaths: List[str],
        validity_mask: Optional[torch.Tensor] = None,
        evidence_quality: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full client forward; returns fused logits and all diagnostics.

        Two MEM compilation passes run: the first obtains decision
        statistics with content-gate priors, the final content gates are
        then recomputed and the concepts recompiled so the correction gate
        matches the final decision distribution. The result dict carries
        the fused/core/anchor logits, reliability signals, gates and the
        morphology regression loss used by local training.
        """
        base_raw = self.frontend.forward_features(images)["pooled"]
        scalar_feats = scalar_feats.to(device=images.device, dtype=images.dtype)
        if validity_mask is None:
            scalar_valid = images.new_ones((images.size(0), 1))
        else:
            scalar_valid = validity_mask.to(device=images.device, dtype=images.dtype).view(
                images.size(0), -1
            )
        scalar_feats_masked = scalar_feats * scalar_valid
        if evidence_quality is None:
            evidence_quality = images.new_ones((images.size(0), 1))
        else:
            evidence_quality = evidence_quality.to(device=images.device, dtype=images.dtype).view(
                images.size(0), -1
            )
        evidence_strength = evidence_quality.clamp(0.0, 1.0)

        # The correction gate and the reliability heads share this anchor call.
        anchor_logits = self.mem.anchor(scalar_feats_masked, scalar_valid)

        branch = self.vsp.adapt(base_raw)
        base_concept = self.vsp.base_concept(branch["morph_feat"])
        compiler_bundle = self._build_compiler_bundle(
            images, scalar_feats_masked, scalar_valid, relpaths
        )
        compiled = self.mem.compile_prior(compiler_bundle, evidence_strength)
        gate_prior = (
            compiler_bundle["local_available"]
            * (0.10 + 0.90 * compiled["compiled_local_reliability"])
            * evidence_strength
        ).clamp(0.0, 1.0)

        # Two compilation passes: the first pass obtains decision statistics
        # with content-gate priors; the final content gates are then recomputed
        # and the concepts recompiled so the correction gate matches the final
        # decision distribution.
        global_context = None
        core_outputs = None
        correction_logits = None
        pass_statistics: Optional[Tuple] = None
        for pass_idx in range(2):
            (
                global_context,
                core_outputs,
                correction_logits,
                pass_statistics,
            ) = self._run_compile_pass(
                compiler_bundle,
                compiled,
                branch,
                base_concept,
                evidence_strength,
                gate_prior,
            )
            if pass_idx == 0:
                _, compiled_conf, compiled_margin, conflict_score = pass_statistics
                final_head_gate = self.mem.content_gate(
                    compiler_bundle["head_reliability"],
                    compiled["head_source_confidence"],
                    compiled_conf,
                    compiled_margin,
                    conflict_score,
                )
                final_tail_gate = self.mem.content_gate(
                    compiler_bundle["tail_reliability"],
                    compiled["tail_source_confidence"],
                    compiled_conf,
                    compiled_margin,
                    conflict_score,
                )
                compiled.update(
                    self.mem.compose_concepts(
                        compiler_bundle=compiler_bundle,
                        metadata_head_concept=compiled["metadata_head_concept"],
                        metadata_tail_concept=compiled["metadata_tail_concept"],
                        head_evidence_concept=compiled["head_evidence_concept"],
                        tail_evidence_concept=compiled["tail_evidence_concept"],
                        head_content_gate=final_head_gate,
                        tail_content_gate=final_tail_gate,
                        local_reliability=compiled["compiled_local_reliability"],
                    )
                )

        _, compiled_conf, compiled_margin, conflict_score = pass_statistics
        core_logits = core_outputs["logits"]
        core_probs = core_outputs["probs"]
        correction_gate = self.mem.compiler_gate(
            compiler_bundle,
            compiled["compiled_local_reliability"],
            evidence_strength,
            compiled_conf,
            compiled_margin,
            conflict_score,
        )
        fused = self._fuse_anchor(
            anchor_logits,
            scalar_feats_masked,
            scalar_valid,
            compiler_bundle,
            compiled,
            branch,
            evidence_strength,
            core_probs,
            core_logits,
            correction_gate,
            correction_logits,
        )

        zero = anchor_logits.new_zeros
        result = {
            "logits": fused["fused_logits"],
            "fusion_logits": fused["fused_logits"],
            "fusion_probs": fused["fused_probs"],
            "core_logits": core_logits,
            "core_probs": core_probs,
            "anchor_logits": fused["anchor_logits"],
            "anchor_probs": fused["anchor_probs"],
            "scalar_feats_masked": scalar_feats_masked,
            "scalar_valid": scalar_valid,
            "base_logits": fused["fused_logits"],
            "base_probs": fused["fused_probs"],
            "private_feature": branch["morph_feat"],
            "base_concept": base_concept,
            "global_context": global_context,
            "head_valid": compiler_bundle["head_valid"],
            "tail_sample_valid": compiler_bundle["tail_sample_valid"],
            "image_quality": compiler_bundle["image_quality"],
            "head_quality": compiler_bundle["head_quality"],
            "tail_quality": compiler_bundle["tail_quality"],
            "local_available": compiler_bundle["local_available"],
            "head_reliability": compiler_bundle["head_reliability"],
            "tail_reliability": compiler_bundle["tail_reliability"],
            "local_reliability": compiled["compiled_local_reliability"],
            "effective_evidence_strength": evidence_strength,
            "correction_gate": correction_gate,
            "compiled_conflict_score": fused["fused_conflict_score"],
            "compiled_morphology_loss": compiled["compiled_morphology_loss"],
            "compiled_head_concept": compiled["compiled_head_concept"],
            "compiled_tail_concept": compiled["compiled_tail_concept"],
            "compiled_head_content_gate": compiled["compiled_head_content_gate"],
            "compiled_tail_content_gate": compiled["compiled_tail_content_gate"],
            "anchor_core_conflict": fused["anchor_core_conflict"],
            "anchor_takeover": fused["anchor_takeover"],
            "anchor_gate": fused["anchor_gate"],
            "rho_m": fused.get("rho_m", zero((anchor_logits.size(0), 1))),
            "rho_v": fused.get("rho_v", zero((anchor_logits.size(0), 1))),
            "rho_m_logits": fused.get("rho_m_logits", zero((anchor_logits.size(0), 1))),
            "rho_v_logits": fused.get("rho_v_logits", zero((anchor_logits.size(0), 1))),
            "rho_gate": fused.get("rho_gate", zero((anchor_logits.size(0), 1))),
            "rho_warmup_alpha": fused.get("rho_warmup_alpha", zero(())),
        }
        return result
