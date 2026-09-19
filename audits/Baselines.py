"""Minimal model wrapper for auditing external federated baselines.

FedAvg, FedBN, FedProx, Ditto, FedProto and Local-only checkpoints produced
by an external baseline codebase can be audited with the same attack
routines. The wrapper exposes the same output dictionary as MorphFLModel so
that the audit code paths are shared.
"""


from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.BoundaryScope import DATASET_CLASS_NAMES
from model.VisualFrontend import PrivateVisualFrontend


class _HierarchicalSMIDSHead(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.ns_gate = nn.Linear(dim, 1)
        self.abn_cls = nn.Linear(dim, 2)


class BaselineModel(nn.Module):
    """Fully fine-tuned DINOv3 with a flat or hierarchical SMIDS classifier."""

    def __init__(
        self,
        pretrained_model_name,
        num_classes,
        class_names,
        full_finetune: bool = True,
        train_norm_layers: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        # Baselines fully fine-tune the backbone (all parameters trainable and
        # exchanged); full_finetune=False restores the legacy frozen+LoRA
        # configuration for old checkpoints.
        self.full_finetune = bool(full_finetune)
        self.backbone_adapter = PrivateVisualFrontend(
            pretrained_model_name=pretrained_model_name,
            full_finetune=self.full_finetune,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            train_norm_layers=train_norm_layers,
        )
        embed_dim = int(self.backbone_adapter.embed_dim)
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        self.is_smids = self.class_names == list(DATASET_CLASS_NAMES)
        if self.is_smids:
            self.classifier_head = _HierarchicalSMIDSHead(embed_dim)
        else:
            self.classifier_head = nn.Linear(embed_dim, self.num_classes)

    def load_from_global_and_private(
        self,
        global_state: Dict[str, torch.Tensor],
        private_state: Optional[Dict[str, torch.Tensor]] = None,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], List[str]]:
        merged = dict(global_state)
        if private_state:
            merged.update(private_state)

        def is_redundant(key: str) -> bool:
            if ".base_layer.weight" in key or ".base_layer.bias" in key:
                base = key.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
                return base in merged
            return False

        filtered = {k: v for k, v in merged.items() if not is_redundant(k)}
        remap: List[Tuple[str, str]] = []
        if self.is_smids:
            for key in ("ns_gate.weight", "ns_gate.bias", "abn_cls.weight", "abn_cls.bias"):
                if key in filtered and f"classifier_head.{key}" not in filtered:
                    remap.append((key, f"classifier_head.{key}"))
        else:
            for key in ("classifier.weight", "classifier.bias"):
                target = "classifier_head." + key[len("classifier."):]
                if key in filtered and target not in filtered:
                    remap.append((key, target))
        for src, dst in remap:
            filtered[dst] = filtered[src]
        if alias_map:
            for src, dst in alias_map.items():
                if src in filtered and dst not in filtered:
                    filtered[dst] = filtered[src]
        return self.load_state_dict(filtered, strict=False)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone_adapter.forward_features(images)
        pooled = features["pooled"]
        if self.is_smids:
            abn_logits = self.classifier_head.abn_cls(pooled)
            ns_logits = self.classifier_head.ns_gate(pooled)
            logits = torch.cat([abn_logits, ns_logits], dim=1)
        else:
            logits = self.classifier_head(pooled)
        probs = F.softmax(logits, dim=1)
        batch_size = pooled.size(0)
        scalar_dim = 2 if self.is_smids else 34
        return {
            "pooled": pooled,
            "hidden_states": features.get("hidden"),
            "base_concept": pooled,
            "global_context": pooled,
            "compiled_head_concept": pooled,
            "compiled_tail_concept": pooled,
            "scalar_feats_masked": torch.zeros(
                batch_size, scalar_dim, dtype=pooled.dtype, device=pooled.device
            ),
            "base_logits": logits,
            "base_probs": probs,
            "fusion_logits": logits,
            "fusion_probs": probs,
        }
