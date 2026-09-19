"""Collect per-sample representations for offline audits."""


from typing import Dict, List

import torch
import torch.nn.functional as F

from utils.Evaluation import forward_batch


def exp_forward(exp: Dict, batch, use_amp: bool = False):
    device = exp["device"]
    if exp.get("variant") == "morphfl":
        return forward_batch(exp["global_model"], batch, device, use_amp=use_amp)
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    if use_amp:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            out = exp["global_model"](images)
    else:
        out = exp["global_model"](images)
    return out, labels


def collect_representations(exp: Dict, max_samples: int = 512) -> Dict[str, torch.Tensor]:
    model = exp["global_model"]
    loader = exp["test_loader"]
    device = exp["device"]
    use_amp = bool(getattr(exp["args"], "amp_enabled", True)) and device.type == "cuda"
    keys = [
        "images",
        "labels",
        "private_feature",
        "base_concept",
        "global_context",
        "compiled_head_concept",
        "compiled_tail_concept",
        "scalar",
        "fusion_probs",
        "anchor_probs",
        "fusion_logits",
        "per_sample_loss_ce",
    ]
    collected: Dict[str, List[torch.Tensor]] = {key: [] for key in keys}
    n = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if n >= max_samples:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out, labels = exp_forward(exp, batch, use_amp=use_amp)
            take = min(labels.size(0), max_samples - n)
            images = batch["image"].to(device, non_blocking=True)
            ce_loss = F.cross_entropy(out["fusion_logits"].float(), labels, reduction="none")
            collected["images"].append(images[:take].float().cpu())
            collected["labels"].append(labels[:take].cpu())
            collected["private_feature"].append(out["private_feature" if "private_feature" in out else "pooled"][:take].float().cpu())
            collected["base_concept"].append(out["base_concept"][:take].float().cpu())
            collected["global_context"].append(out["global_context"][:take].float().cpu())
            collected["compiled_head_concept"].append(out["compiled_head_concept"][:take].float().cpu())
            collected["compiled_tail_concept"].append(out["compiled_tail_concept"][:take].float().cpu())
            collected["scalar"].append(out["scalar_feats_masked"][:take].float().cpu())
            collected["fusion_probs"].append(out["fusion_probs"][:take].float().cpu())
            collected["anchor_probs"].append(
                out.get("anchor_probs", out["fusion_probs"])[:take].float().cpu()
            )
            collected["fusion_logits"].append(out["fusion_logits"][:take].float().cpu())
            collected["per_sample_loss_ce"].append(ce_loss[:take].float().cpu())
            n += take
    for key in list(collected.keys()):
        collected[key] = torch.cat(collected[key], dim=0) if collected[key] else torch.tensor([])
    return collected
