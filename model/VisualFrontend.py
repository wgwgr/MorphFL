"""Private visual frontend (paper, Private Visual Frontend).

Everything in this module stays client-local and is never aggregated:

- a DINOv3 ViT-B/16 encoder producing the CLS-token embedding. MorphFL keeps
  it frozen with small private LoRA adapters (a local training-cost
  optimization); audited baselines construct it in full-fine-tuning mode;
- optionally, trainable local normalization layers (used by FedBN-style
  configurations).

The shared rank-16/64-d projections that consume the encoder embedding live
in :mod:`morphfl.model.VSP` (the VSP).
"""


import os
from typing import Dict, List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel


def resolve_backbone_name(name: str) -> str:
    """Resolve a backbone name against local workspace weight directories."""
    if os.path.exists(name):
        return name
    here = os.path.dirname(os.path.abspath(__file__))
    # Search the repo root and the enclosing workspace root (weights may live
    # one level above the repository), plus a DINOV3/ subdirectory in either.
    search_roots = [
        os.path.abspath(os.path.join(here, "..")),       # repository root
        os.path.abspath(os.path.join(here, "..", "..")),  # workspace root
    ]
    for root in search_roots:
        for suffix in ("", "DINOV3"):
            candidate = os.path.join(root, suffix, name) if suffix else os.path.join(root, name)
            if os.path.exists(candidate):
                return candidate
    return name


def _detect_lora_targets(backbone: nn.Module) -> List[str]:
    """Auto-detect the attention projection modules to adapt with LoRA."""
    module_names = [name for name, _ in backbone.named_modules()]
    if any(name.endswith("q_proj") for name in module_names):
        return ["q_proj", "v_proj"]
    if any(name.endswith("query") for name in module_names):
        return ["query", "value"]
    return []


class PrivateVisualFrontend(nn.Module):
    """Frozen DINOv3 ViT-B/16 with private LoRA(rank 8) adapters."""

    def __init__(
        self,
        pretrained_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        train_norm_layers: bool = False,
        full_finetune: bool = False,
        attn_implementation: str = "sdpa",
    ):
        """Load the frozen encoder and attach private LoRA/norm adapters."""
        super().__init__()
        self.resolved_name = resolve_backbone_name(pretrained_model_name)
        self.backbone_type = "vit"
        self.full_finetune = bool(full_finetune)
        self.requested_lora = (not self.full_finetune) and int(lora_r) > 0
        self.use_lora = False
        self.lora_target_modules: List[str] = []
        self.tuning_mode = "full_finetune" if self.full_finetune else "frozen"
        self.train_norm_layers = bool(train_norm_layers)

        load_kwargs: Dict = {}
        if "dinov3" in self.resolved_name.lower():
            # SDPA is mathematically equivalent to eager under bf16 autocast
            # and skips materializing the attention matrix.
            load_kwargs["attn_implementation"] = str(attn_implementation)
        backbone = AutoModel.from_pretrained(self.resolved_name, **load_kwargs)
        self.embed_dim = int(backbone.config.hidden_size)

        # Full fine-tuning: every backbone parameter is locally trainable.
        # Otherwise the encoder is frozen and only (optionally) LoRA-adapted.
        if self.full_finetune:
            for param in backbone.parameters():
                param.requires_grad = True
        else:
            for param in backbone.parameters():
                param.requires_grad = False

        if self.requested_lora:
            self.lora_target_modules = _detect_lora_targets(backbone)
            if self.lora_target_modules:
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=self.lora_target_modules,
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                backbone = get_peft_model(backbone, lora_config)
                self.use_lora = True
                self.tuning_mode = "lora"

        if self.train_norm_layers:
            train_norm_count = 0
            for name, param in backbone.named_parameters():
                lower = name.lower()
                if "norm" in lower or ".bn" in lower or lower.startswith("bn"):
                    param.requires_grad = True
                    train_norm_count += 1
            if train_norm_count > 0:
                self.tuning_mode = "lora+local-norm" if self.use_lora else "local-norm"

        self.backbone = backbone

    def forward_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the encoder and return the CLS embedding plus all token states."""
        outputs = self.backbone(pixel_values=images)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0, :]
        return {"pooled": pooled, "hidden": hidden}
