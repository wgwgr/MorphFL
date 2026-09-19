"""Client-local optimization (the local update of the federated loop)."""

import math
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from utils.Evaluation import TRAIN_MEAN_KEYS, TRAIN_SUM_KEYS, forward_batch


def _linear_value(
    current_round: int,
    total_rounds: int,
    start_value: float,
    end_value: float,
    decay_start_round: int,
) -> float:
    """Linearly interpolate from start to end after ``decay_start_round``."""
    if start_value <= 0:
        return 0.0
    total_rounds = max(int(total_rounds), 1)
    current_round = min(max(int(current_round), 1), total_rounds)
    decay_start_round = min(max(int(decay_start_round), 1), total_rounds)
    if current_round <= decay_start_round or total_rounds <= decay_start_round:
        return float(start_value)
    progress = float(current_round - decay_start_round) / float(total_rounds - decay_start_round)
    return float(start_value + (end_value - start_value) * progress)


def _cosine_value(
    current_round: int,
    total_rounds: int,
    start_value: float,
    end_value: float,
    decay_start_round: int,
) -> float:
    """Cosine-anneal from start to end after ``decay_start_round``."""
    if start_value <= 0:
        return 0.0
    total_rounds = max(int(total_rounds), 1)
    current_round = min(max(int(current_round), 1), total_rounds)
    decay_start_round = min(max(int(decay_start_round), 1), total_rounds)
    if current_round <= decay_start_round or total_rounds <= decay_start_round:
        return float(start_value)
    progress = float(current_round - decay_start_round) / float(max(total_rounds - decay_start_round, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return float(end_value + (start_value - end_value) * cosine)


def current_round_lr_scale(args) -> float:
    """Per-round learning-rate scale under the configured schedule shape."""
    schedule = str(getattr(args, "round_lr_schedule", "linear") or "linear").strip().lower()
    kwargs = dict(
        current_round=int(getattr(args, "_current_round", 1)),
        total_rounds=int(getattr(args, "rounds", 20)),
        start_value=float(getattr(args, "round_lr_scale_start", 1.0)),
        end_value=float(getattr(args, "round_lr_scale_end", 0.50)),
        decay_start_round=int(getattr(args, "round_lr_decay_start_round", 6)),
    )
    if schedule == "cosine":
        return _cosine_value(**kwargs)
    return _linear_value(**kwargs)


def _build_param_groups(model, args):
    """Group trainable parameters into head/backbone/LoRA AdamW groups.

    The backbone and LoRA groups receive their configured LR scales, and
    every group is multiplied by the current per-round schedule value.
    """
    round_scale = max(float(current_round_lr_scale(args)), 0.0)
    head_params, backbone_params, lora_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif name.startswith("frontend.backbone.") or name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * round_scale, "weight_decay": args.weight_decay})
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": args.lr * args.backbone_lr_scale * round_scale,
            "weight_decay": args.weight_decay,
        })
    if lora_params:
        groups.append({
            "params": lora_params,
            "lr": args.lr * args.lora_lr_scale * round_scale,
            "weight_decay": args.weight_decay,
        })
    return groups


def train_local(
    model,
    loader,
    args,
    device,
    local_class_prior: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Run one client's local update over ``loader`` for ``local_epochs``.

    The joint loss is the fused CE (optionally with local-prior logit
    adjustment and label smoothing) plus the morphology regression, the
    anchor CE and the URC rho BCE terms. During prewarm rounds only the
    scalar-only anchor branch is trained; after each joint step the anchor
    receives extra scalar-only micro-steps. Returns the mean loss/accuracy
    plus per-batch diagnostic means (one entry per TRAIN metric key).
    """
    start = time.perf_counter()
    model.train()
    if hasattr(model, "set_round_context"):
        model.set_round_context(int(getattr(args, "_current_round", 1)), int(getattr(args, "rounds", 20)))

    optimizer = AdamW(_build_param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_seen = 0
    zero = lambda: torch.zeros((), device=device)
    metric_sums = {key: zero() for key, _ in (*TRAIN_SUM_KEYS, *TRAIN_MEAN_KEYS)}
    metric_sums["round_lr_scale_current"] = zero()
    # bf16 has the dynamic range required by the DINOv3 eager attention;
    # gradient scaling is unnecessary with bf16.
    use_amp = bool(getattr(args, "amp_enabled", True)) and device.type == "cuda"

    # Local-prior logit adjustment for class-skewed clients (MBP-safe: the
    # prior is derived from client-local label statistics only).
    class_balance_mode = str(getattr(args, "class_balance_mode", "none") or "none").strip().lower()
    class_balance_tau = float(getattr(args, "class_balance_tau", 1.0))
    prior_bias: Optional[torch.Tensor] = None
    if class_balance_mode == "logit_adjust" and local_class_prior is not None and class_balance_tau > 0:
        prior_bias = torch.log(local_class_prior.to(device=device).clamp_min(1e-8)) * class_balance_tau
    label_smoothing = min(max(float(getattr(args, "label_smoothing", 0.0)), 0.0), 0.9)

    for _ in range(args.local_epochs):
        for batch in loader:
            # Anchor warm-up: the anchor reads only morphology scalars and is
            # independent of the visual backbone, so it is pre-trained cheaply
            # before the joint phase.
            prewarm_rounds = int(getattr(args, "anchor_prewarm_rounds", 2))
            if prewarm_rounds > 0 and int(getattr(args, "_current_round", 1)) <= prewarm_rounds:
                pre_steps = int(getattr(args, "anchor_prewarm_micro_steps", 6))
                weight = float(getattr(args, "anchor_supervision_weight", 2.0))
                non_blocking = device.type == "cuda"
                fp32 = torch.float32
                labels = batch["label"].to(device, non_blocking=non_blocking)
                scalars = batch["scalar_feats"].to(device, non_blocking=non_blocking, dtype=fp32)
                valid = batch["validity_mask"].to(device, non_blocking=non_blocking, dtype=fp32).view(-1, 1)
                masked = scalars * valid
                for _ in range(pre_steps):
                    optimizer.zero_grad(set_to_none=True)
                    anchor_input = torch.cat([model.mem.scalar_norm(masked), valid], dim=1)
                    loss = weight * F.cross_entropy(model.mem.anchor_classifier(anchor_input), labels)
                    loss.backward()
                    optimizer.step()
                continue

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out, labels = forward_batch(model, batch, device, use_amp=use_amp)

                fused_logits = out["base_logits"]
                if prior_bias is not None:
                    fused_logits = fused_logits + prior_bias.to(fused_logits.dtype)
                loss = float(getattr(args, "base_loss_weight", 1.0)) * F.cross_entropy(
                    fused_logits, labels, label_smoothing=label_smoothing
                )
                compiler_weight = float(getattr(args, "compiler_supervision_weight", 0.25))
                if compiler_weight > 0:
                    loss = loss + compiler_weight * out["compiled_morphology_loss"]

                anchor_weight = float(getattr(args, "anchor_supervision_weight", 2.0))
                if anchor_weight > 0:
                    loss = loss + anchor_weight * F.cross_entropy(out["anchor_logits"], labels)

                rho_weight = float(getattr(args, "rho_loss_weight", 0.15))
                rho_loss = out["base_logits"].new_zeros(())
                if rho_weight > 0 and bool(getattr(args, "rho_enabled", False)):
                    rho_m_logits = out["rho_m_logits"].reshape(-1).float()
                    rho_v_logits = out["rho_v_logits"].reshape(-1).float()
                    anchor_correct = (out["anchor_probs"].argmax(dim=1) == labels).to(rho_m_logits.dtype)
                    core_correct = (out["core_probs"].argmax(dim=1) == labels).to(rho_v_logits.dtype)
                    rho_loss = F.binary_cross_entropy_with_logits(
                        rho_m_logits, anchor_correct.detach()
                    ) + F.binary_cross_entropy_with_logits(rho_v_logits, core_correct.detach())
                    loss = loss + rho_weight * rho_loss
                out["rho_loss"] = rho_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Anchor micro-steps: the anchor MLP is tiny and independent of
            # the backbone, so extra scalar-only updates improve it at low cost.
            micro_steps = int(getattr(args, "anchor_micro_steps", 4))
            if micro_steps > 0:
                fp32 = torch.float32
                for _ in range(micro_steps):
                    optimizer.zero_grad(set_to_none=True)
                    anchor_input = torch.cat(
                        [
                            model.mem.scalar_norm(out["scalar_feats_masked"].float()),
                            out["scalar_valid"].float(),
                        ],
                        dim=1,
                    )
                    anchor_loss = anchor_weight * F.cross_entropy(
                        model.mem.anchor_classifier(anchor_input), labels
                    )
                    anchor_loss.backward()
                    optimizer.step()

            batch_size = int(labels.size(0))
            total_loss += loss.detach() * batch_size
            total_correct += (out["logits"].argmax(dim=1) == labels).sum().detach()
            total_seen += batch_size

            for key, out_key in (*TRAIN_SUM_KEYS, *TRAIN_MEAN_KEYS):
                if out_key in out:
                    tensor = out[out_key]
                    value = tensor.mean().detach() if tensor.ndim > 0 else tensor.detach()
                    metric_sums[key] += value * batch_size
            metric_sums["round_lr_scale_current"] += metric_sums["round_lr_scale_current"].new_tensor(
                current_round_lr_scale(args) * batch_size
            )

    metrics = {
        "loss": float((total_loss / max(total_seen, 1)).detach().cpu().item()),
        "acc": float((total_correct / max(total_seen, 1)).detach().cpu().item()),
        "train_time_sec": float(time.perf_counter() - start),
    }
    for key, value in metric_sums.items():
        metrics[key] = float((value / max(total_seen, 1)).detach().cpu().item())
    return metrics
