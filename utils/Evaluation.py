"""Batch forwarding, global/personalized evaluation and metrics serialization."""

import csv
import os
from statistics import mean
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

from utils.BoundaryScope import CLASS_DISPLAY_NAMES, resolve_scope, summarize_server_visibility
from utils.Utils import compute_metrics_multiclass, summarize_multiclass_metric_dicts
from utils.Utils import write_json


def forward_batch(model, batch, device, use_amp: bool = False):
    """Move one collated batch to the device and run the model forward."""
    non_blocking = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32
    images = batch["image"].to(device, non_blocking=non_blocking, dtype=dtype)
    labels = batch["label"].to(device, non_blocking=non_blocking)
    scalar_feats = batch["scalar_feats"].to(device, non_blocking=non_blocking, dtype=dtype)
    validity_mask = batch["validity_mask"].to(device, non_blocking=non_blocking, dtype=dtype)
    evidence_quality = batch.get("evidence_quality")
    out = model(
        images,
        scalar_feats,
        batch["relpath"],
        validity_mask=validity_mask,
        evidence_quality=(
            evidence_quality.to(device, non_blocking=non_blocking, dtype=dtype)
            if evidence_quality is not None
            else None
        ),
    )
    return out, labels


_SUM_KEYS = (
    ("local_available", "local_available"),
    ("image_quality", "image_quality"),
    ("head_quality", "head_quality"),
    ("tail_quality", "tail_quality"),
    ("head_reliability", "head_reliability"),
    ("tail_reliability", "tail_reliability"),
    ("local_reliability", "local_reliability"),
)
_MEAN_KEYS = (
    ("compiled_conflict_score", "compiled_conflict_score"),
    ("anchor_core_conflict", "anchor_core_conflict"),
    ("anchor_takeover", "anchor_takeover"),
    ("anchor_gate", "anchor_gate"),
    ("rho_m_mean", "rho_m"),
    ("rho_v_mean", "rho_v"),
    ("rho_gate_mean", "rho_gate"),
    ("rho_warmup_alpha", "rho_warmup_alpha"),
    ("correction_gate_mean", "correction_gate"),
    ("compiled_head_content_gate", "compiled_head_content_gate"),
    ("compiled_tail_content_gate", "compiled_tail_content_gate"),
    ("compiled_morphology_loss", "compiled_morphology_loss"),
    ("effective_evidence_strength_mean", "effective_evidence_strength"),
)

# Per-batch metric accumulation keys shared with utils/LocalTraining.py:
# (metric_key, out_key). LocalTraining additionally tracks "round_lr_scale_current".
TRAIN_SUM_KEYS = (
    ("image_quality", "image_quality"),
    ("head_quality", "head_quality"),
    ("tail_quality", "tail_quality"),
    ("head_reliability", "head_reliability"),
    ("tail_reliability", "tail_reliability"),
    ("local_reliability", "local_reliability"),
    ("head_valid_rate", "head_valid"),
    ("tail_valid_rate", "tail_sample_valid"),
)
TRAIN_MEAN_KEYS = (
    ("compiled_morphology_loss", "compiled_morphology_loss"),
    ("compiled_conflict_score", "compiled_conflict_score"),
    ("anchor_core_conflict", "anchor_core_conflict"),
    ("anchor_takeover", "anchor_takeover"),
    ("anchor_gate", "anchor_gate"),
    ("rho_m_mean", "rho_m"),
    ("rho_v_mean", "rho_v"),
    ("rho_gate_mean", "rho_gate"),
    ("rho_loss", "rho_loss"),
    ("correction_gate_mean", "correction_gate"),
    ("compiled_head_content_gate", "compiled_head_content_gate"),
    ("compiled_tail_content_gate", "compiled_tail_content_gate"),
    ("effective_evidence_strength_mean", "effective_evidence_strength"),
)


def _collect_scalars(out: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
    """Stack the per-batch diagnostic scalars tracked by the key tables."""
    scalars = [out[key].sum().detach() for _, key in _SUM_KEYS]
    for _, key in _MEAN_KEYS:
        if key in out:
            tensor = out[key]
            scalars.append(tensor.mean().detach() if tensor.ndim > 0 else tensor.detach())
    return scalars


def evaluate_model(model, loader, device, args=None, return_outputs: bool = False):
    """Evaluate one model over a loader; returns fused/core/anchor metrics.

    ``return_outputs=True`` additionally returns raw targets, predictions
    and probabilities for pooled or per-client analyses.
    """
    model.eval()
    fusion_preds, fusion_probs = [], []
    core_preds, core_probs = [], []
    anchor_preds, anchor_probs = [], []
    targets_list = []
    total_seen = 0
    sum_values = {key: 0.0 for key, _ in _SUM_KEYS}
    mean_sums = {key: 0.0 for key, _ in _MEAN_KEYS}
    mean_counts = {key: 0 for key, _ in _MEAN_KEYS}

    use_amp = bool(getattr(args, "amp_enabled", True)) and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out, labels = forward_batch(model, batch, device, use_amp=use_amp)
            batch_size = int(labels.size(0))

            fusion_probs.append(out["fusion_probs"].cpu().numpy())
            core_probs.append(out["core_probs"].cpu().numpy())
            anchor_probs.append(out["anchor_probs"].cpu().numpy())
            fusion_preds.append(out["fusion_probs"].argmax(dim=1).cpu().numpy())
            core_preds.append(out["core_probs"].argmax(dim=1).cpu().numpy())
            anchor_preds.append(out["anchor_probs"].argmax(dim=1).cpu().numpy())
            targets_list.append(labels.cpu().numpy())
            total_seen += batch_size

            scalars = torch.stack(_collect_scalars(out)).cpu()
            cursor = 0
            for key, _ in _SUM_KEYS:
                sum_values[key] += float(scalars[cursor].item())
                cursor += 1
            for key, out_key in _MEAN_KEYS:
                if out_key in out:
                    mean_sums[key] += float(scalars[cursor].item()) * batch_size
                    mean_counts[key] += batch_size
                    cursor += 1

    targets = np.concatenate(targets_list)
    fusion_preds = np.concatenate(fusion_preds)
    fusion_probs = np.concatenate(fusion_probs)
    core_preds = np.concatenate(core_preds)
    core_probs = np.concatenate(core_probs)
    anchor_preds = np.concatenate(anchor_preds)
    anchor_probs = np.concatenate(anchor_probs)
    class_names = list(getattr(model, "class_names", CLASS_DISPLAY_NAMES))

    metrics = compute_metrics_multiclass(fusion_preds, targets, fusion_probs, class_names=class_names)
    core_metrics = compute_metrics_multiclass(core_preds, targets, core_probs, class_names=class_names)
    anchor_metrics = compute_metrics_multiclass(
        anchor_preds, targets, anchor_probs, class_names=class_names
    )

    metrics["local_available_rate"] = sum_values["local_available"] / max(total_seen, 1)
    metrics["image_quality_mean"] = sum_values["image_quality"] / max(total_seen, 1)
    metrics["head_quality_mean"] = sum_values["head_quality"] / max(total_seen, 1)
    metrics["tail_quality_mean"] = sum_values["tail_quality"] / max(total_seen, 1)
    metrics["head_reliability_mean"] = sum_values["head_reliability"] / max(total_seen, 1)
    metrics["tail_reliability_mean"] = sum_values["tail_reliability"] / max(total_seen, 1)
    metrics["local_reliability_mean"] = sum_values["local_reliability"] / max(total_seen, 1)
    for key, total in mean_sums.items():
        if mean_counts[key] > 0:
            metrics[key] = total / mean_counts[key]
    metrics["core_macro_f1"] = core_metrics["macro_f1"]
    metrics["anchor_macro_f1"] = anchor_metrics["macro_f1"]
    metrics["core_branch"] = core_metrics
    metrics["anchor_branch"] = anchor_metrics

    if not return_outputs:
        return metrics
    outputs = {
        "targets": targets,
        "fusion_preds": fusion_preds,
        "fusion_probs": fusion_probs,
        "core_preds": core_preds,
        "core_probs": core_probs,
        "anchor_preds": anchor_preds,
        "anchor_probs": anchor_probs,
    }
    return metrics, outputs


def weighted_row_metric_mean(rows, key: str):
    """Example-weighted mean of one metric across per-client rows."""
    total, examples = 0.0, 0
    for row in rows:
        num_examples = int(row.get("num_examples", 0))
        if key in row and num_examples > 0:
            total += float(row[key]) * num_examples
            examples += num_examples
    return float(total / examples) if examples > 0 else None


_AGGREGATE_KEYS = [
    "core_macro_f1",
    "anchor_macro_f1",
    "local_available_rate",
    "image_quality_mean",
    "head_quality_mean",
    "tail_quality_mean",
    "head_reliability_mean",
    "tail_reliability_mean",
    "local_reliability_mean",
    "compiled_conflict_score",
    "anchor_core_conflict",
    "anchor_takeover",
    "anchor_gate",
    "rho_m_mean",
    "rho_v_mean",
    "rho_gate_mean",
    "rho_warmup_alpha",
    "correction_gate_mean",
    "compiled_head_content_gate",
    "compiled_tail_content_gate",
    "compiled_morphology_loss",
    "effective_evidence_strength_mean",
]


def _summary_block(rows, class_names, fusion_metrics, protocol: str):
    """Assemble the aggregate block shared by both evaluation protocols."""
    summary = summarize_multiclass_metric_dicts(rows, class_names=class_names)
    block = {
        "evaluation_protocol": protocol,
        "acc": float(fusion_metrics.get("acc", 0.0)),
        "balanced_acc": float(fusion_metrics.get("balanced_acc", 0.0)),
        "macro_f1": float(fusion_metrics.get("macro_f1", 0.0)),
        "macro_auc": float(fusion_metrics.get("macro_auc", 0.0)),
        "macro_precision": float(fusion_metrics.get("macro_precision", 0.0)),
        "macro_recall": float(fusion_metrics.get("macro_recall", 0.0)),
        "weighted_f1": float(fusion_metrics.get("weighted_f1", 0.0)),
        "per_class_summary": fusion_metrics.get("per_class", {}),
        "per_client": rows,
        "num_clients_evaluated": int(len(rows)),
        "client_metric_std": {
            key: summary["overall"].get(key, {}).get("std", 0.0)
            for key in ("acc", "balanced_acc", "macro_f1", "macro_auc", "weighted_f1")
        },
    }
    if protocol == "shifted_global_concat":
        block["num_examples_evaluated"] = int(fusion_metrics.get("total_samples", 0))
    for key in _AGGREGATE_KEYS:
        value = weighted_row_metric_mean(rows, key)
        if value is None and key in fusion_metrics:
            value = fusion_metrics.get(key)
        if value is not None:
            block[key] = float(value)
    return block


def evaluate_personalized_models_shifted_concat(models, test_loaders, device, args=None):
    """Evaluate each personalized model on its shifted loader and pool predictions.

    The aggregated metrics are computed over the concatenation of all
    per-client predictions instead of averaging per-client metrics.
    """
    rows = []
    first_model = next(iter(models.values()))
    class_names = list(getattr(first_model, "class_names", CLASS_DISPLAY_NAMES))
    all_targets, all_preds, all_probs = [], [], []
    for client_name, model in models.items():
        metrics, outputs = evaluate_model(
            model, test_loaders[client_name], device, args=args, return_outputs=True
        )
        row = dict(metrics)
        row["client"] = client_name
        row["num_examples"] = int(outputs["targets"].shape[0])
        rows.append(row)
        all_targets.append(outputs["targets"])
        all_preds.append(outputs["fusion_preds"])
        all_probs.append(outputs["fusion_probs"])
    fusion_metrics = compute_metrics_multiclass(
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
        class_names=class_names,
    )
    return _summary_block(rows, class_names, fusion_metrics, "shifted_global_concat")


def evaluate_personalized_models(models, test_loaders, device, args=None):
    """Evaluate each personalized model and average the per-client metrics."""
    rows = []
    first_model = next(iter(models.values()))
    class_names = list(getattr(first_model, "class_names", CLASS_DISPLAY_NAMES))
    for client_name, model in models.items():
        loader = test_loaders[client_name] if isinstance(test_loaders, dict) else test_loaders
        metrics = evaluate_model(model, loader, device, args=args)
        row = dict(metrics)
        row["client"] = client_name
        row["num_examples"] = int(len(loader.dataset))
        rows.append(row)

    summary = summarize_multiclass_metric_dicts(rows, class_names=class_names)
    overall = {
        "evaluation_protocol": "per_client_mean",
        "acc": summary["overall"].get("acc", {}).get("mean", 0.0),
        "balanced_acc": summary["overall"].get("balanced_acc", {}).get("mean", 0.0),
        "macro_f1": summary["overall"].get("macro_f1", {}).get("mean", 0.0),
        "macro_auc": summary["overall"].get("macro_auc", {}).get("mean", 0.0),
        "macro_precision": summary["overall"].get("macro_precision", {}).get("mean", 0.0),
        "macro_recall": summary["overall"].get("macro_recall", {}).get("mean", 0.0),
        "weighted_f1": summary["overall"].get("weighted_f1", {}).get("mean", 0.0),
        "per_class_summary": summary["per_class"],
        "per_client": rows,
        "num_clients_evaluated": int(len(rows)),
        "client_metric_std": {
            key: summary["overall"].get(key, {}).get("std", 0.0)
            for key in ("acc", "balanced_acc", "macro_f1", "macro_auc", "weighted_f1")
        },
    }
    for key in _AGGREGATE_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            overall[key] = float(mean(values))
    return overall


def _summarize_rows_by_factor(rows, client_domain_profiles: Mapping[str, object]):
    """Group per-client rows by domain-shift factor and summarize means."""
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        client_name = str(row.get("client", ""))
        profile = client_domain_profiles.get(client_name, {})
        profile_name = str(profile.get("name", "")) if isinstance(profile, Mapping) else ""
        factor = profile_name.split("_", 1)[0] if profile_name else "clean"
        grouped.setdefault(factor, []).append(row)
    return {
        factor: {
            "num_clients": int(len(factor_rows)),
            "num_examples": int(sum(int(row.get("num_examples", 0)) for row in factor_rows)),
            "macro_f1_mean": float(np.mean([float(row.get("macro_f1", 0.0)) for row in factor_rows])),
            "acc_mean": float(np.mean([float(row.get("acc", 0.0)) for row in factor_rows])),
            "balanced_acc_mean": float(
                np.mean([float(row.get("balanced_acc", 0.0)) for row in factor_rows])
            ),
        }
        for factor, factor_rows in grouped.items()
    }


def attach_shift_breakdown(global_test: Dict[str, object], client_domain_profiles) -> Dict[str, object]:
    """Enrich an evaluation block with worst-case client and per-factor rows."""
    enriched = dict(global_test)
    rows = [row for row in enriched.get("per_client", []) if isinstance(row, Mapping)]
    if rows:
        worst = min(rows, key=lambda row: float(row.get("macro_f1", 0.0)))
        enriched["worst_case"] = {
            "client": worst.get("client"),
            "macro_f1": float(worst.get("macro_f1", 0.0)),
            "acc": float(worst.get("acc", 0.0)),
            "balanced_acc": float(worst.get("balanced_acc", 0.0)),
        }
        enriched["per_factor"] = _summarize_rows_by_factor(rows, client_domain_profiles)
    return enriched


def summarize_timing(per_round_logs, total_train_time_sec, started_at, finished_at):
    """Aggregate per-round timing rows into run-level timing statistics."""
    round_times = [float(row.get("timing", {}).get("round_time_sec", 0.0)) for row in per_round_logs]
    train_times = [float(row.get("timing", {}).get("train_time_sec_sum", 0.0)) for row in per_round_logs]
    eval_times = [float(row.get("timing", {}).get("global_eval_time_sec", 0.0)) for row in per_round_logs]
    agg_times = [float(row.get("timing", {}).get("aggregation_time_sec", 0.0)) for row in per_round_logs]
    mean = lambda values: float(np.mean(values)) if values else 0.0
    std = lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "total_train_time_sec": float(total_train_time_sec),
        "num_rounds": int(len(per_round_logs)),
        "avg_round_time_sec": mean(round_times),
        "round_time_std_sec": std(round_times),
        "avg_train_time_sec_sum": mean(train_times),
        "avg_eval_time_sec": mean(eval_times),
        "avg_aggregation_time_sec": mean(agg_times),
    }


def write_round_log_artifacts(output_dir: str, per_round_logs) -> Dict[str, str]:
    """Persist round logs as JSON plus a flat per-round metrics CSV.

    Rounds with skipped test evaluations (``--test-eval-every-round``) fall
    back to zeros/client-row means for the test-derived CSV columns.
    """
    round_logs_path = os.path.join(output_dir, "round_logs.json")
    write_json(round_logs_path, list(per_round_logs))

    csv_path = os.path.join(output_dir, "round_metrics.csv")
    fieldnames = [
        "round",
        "val_macro_f1",
        "val_acc",
        "macro_f1",
        "acc",
        "balanced_acc",
        "worst_case_macro_f1",
        "worst_case_acc",
        "core_macro_f1",
        "anchor_macro_f1",
        "local_available_rate",
        "local_reliability_mean",
        "effective_evidence_strength_mean",
        "compiled_conflict_score",
        "anchor_core_conflict",
        "anchor_takeover",
        "anchor_gate",
        "rho_m_mean",
        "rho_v_mean",
        "rho_gate_mean",
        "rho_warmup_alpha",
        "correction_gate_mean",
        "compiled_head_content_gate",
        "compiled_tail_content_gate",
        "compiled_morphology_loss",
        "train_time_sec_sum",
        "aggregation_time_sec",
        "global_eval_time_sec",
        "val_eval_time_sec",
        "round_time_sec",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_round_logs:
            timing = row.get("timing", {})
            global_test = row.get("global_test", {})
            global_val = row.get("global_val", {})
            worst_case = global_test.get("worst_case", {}) if isinstance(global_test, Mapping) else {}
            client_rows = []
            for client_name, client_log in (row.get("clients", {}) or {}).items():
                if isinstance(client_log, Mapping):
                    client_row = dict(client_log)
                    client_row["client"] = client_name
                    client_row["num_examples"] = int(client_log.get("train_size", 0))
                    client_rows.append(client_row)

            def _metric(name: str) -> float:
                """Prefer the round's test metric; fall back to client rows."""
                if name in global_test:
                    return float(global_test[name])
                return float(weighted_row_metric_mean(client_rows, name) or 0.0)

            writer.writerow({
                "round": int(row.get("round", 0)),
                "val_macro_f1": float(global_val.get("macro_f1", 0.0)) if isinstance(global_val, Mapping) else 0.0,
                "val_acc": float(global_val.get("acc", 0.0)) if isinstance(global_val, Mapping) else 0.0,
                "macro_f1": float(global_test.get("macro_f1", 0.0)),
                "acc": float(global_test.get("acc", 0.0)),
                "balanced_acc": float(global_test.get("balanced_acc", 0.0)),
                "worst_case_macro_f1": float(worst_case.get("macro_f1", 0.0)),
                "worst_case_acc": float(worst_case.get("acc", 0.0)),
                "core_macro_f1": float(global_test.get("core_macro_f1", 0.0)),
                "anchor_macro_f1": float(global_test.get("anchor_macro_f1", 0.0)),
                "local_available_rate": float(global_test.get("local_available_rate", 0.0)),
                "local_reliability_mean": float(global_test.get("local_reliability_mean", 0.0)),
                "effective_evidence_strength_mean": _metric("effective_evidence_strength_mean"),
                "compiled_conflict_score": _metric("compiled_conflict_score"),
                "anchor_core_conflict": _metric("anchor_core_conflict"),
                "anchor_takeover": _metric("anchor_takeover"),
                "anchor_gate": _metric("anchor_gate"),
                "rho_m_mean": _metric("rho_m_mean"),
                "rho_v_mean": _metric("rho_v_mean"),
                "rho_gate_mean": _metric("rho_gate_mean"),
                "rho_warmup_alpha": _metric("rho_warmup_alpha"),
                "correction_gate_mean": _metric("correction_gate_mean"),
                "compiled_head_content_gate": _metric("compiled_head_content_gate"),
                "compiled_tail_content_gate": _metric("compiled_tail_content_gate"),
                "compiled_morphology_loss": _metric("compiled_morphology_loss"),
                "train_time_sec_sum": float(timing.get("train_time_sec_sum", 0.0)),
                "aggregation_time_sec": float(timing.get("aggregation_time_sec", 0.0)),
                "global_eval_time_sec": float(timing.get("global_eval_time_sec", 0.0)),
                "val_eval_time_sec": float(timing.get("val_eval_time_sec", 0.0)),
                "round_time_sec": float(timing.get("round_time_sec", 0.0)),
            })
    return {"round_logs_json": round_logs_path, "round_metrics_csv": csv_path}


def strip_metric_keys(payload: object, drop_keys: Iterable[str]) -> object:
    """Recursively remove metric keys from nested metric payloads."""
    drop = set(drop_keys)
    if isinstance(payload, Mapping):
        return {k: strip_metric_keys(v, drop) for k, v in payload.items() if k not in drop}
    if isinstance(payload, list):
        return [strip_metric_keys(item, drop) for item in payload]
    return payload


def build_result_header(args, split_obj, global_state, shared_keys, private_keys):
    """Build the experiment header: protocol, model config and visibility."""
    scope = resolve_scope(args)
    visibility = summarize_server_visibility(
        global_state,
        shared_keys,
        private_keys,
        rounds=int(args.rounds),
        num_clients=len(split_obj.get("clients", {})),
    )
    return {
        "method": args.method,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "shared_scope": scope["shared_scope"],
        "shared_scope_mode": scope["shared_scope_mode"],
        "private_scope": scope["private_scope"],
        "selection_metric": "macro_f1",
        "shared_aggregation": {"strategy": "sample_size"},
        "train_norm_layers": bool(getattr(args, "train_norm_layers", False)),
        "morphfl": {
            "concept_dim": int(getattr(args, "concept_dim", 64)),
            "morph_adapter_rank": int(getattr(args, "morph_adapter_rank", 16)),
            "morph_adapter_alpha": float(getattr(args, "morph_adapter_alpha", 16.0)),
            "adapter_dropout": float(getattr(args, "adapter_dropout", 0.1)),
            "hidden_dim": int(getattr(args, "hidden_dim", 256)),
            "share_global_context": bool(getattr(args, "share_global_context", True)),
            "style_residual_scale": float(getattr(args, "style_residual_scale", 0.12)),
            "style_logit_scale": float(getattr(args, "style_logit_scale", 0.05)),
            "style_delta_max_norm": float(getattr(args, "style_delta_max_norm", 0.60)),
            "compiler_supervision_weight": float(getattr(args, "compiler_supervision_weight", 0.25)),
            "anchor_supervision_weight": float(getattr(args, "anchor_supervision_weight", 2.0)),
            "anchor_micro_steps": int(getattr(args, "anchor_micro_steps", 4)),
            "anchor_prewarm_rounds": int(getattr(args, "anchor_prewarm_rounds", 2)),
            "rho_enabled": bool(getattr(args, "rho_enabled", False)),
            "rho_warmup_rounds": int(getattr(args, "rho_warmup_rounds", 4)),
            "rho_loss_weight": float(getattr(args, "rho_loss_weight", 0.15)),
            "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
            "class_balance_mode": str(getattr(args, "class_balance_mode", "none")),
            "class_balance_tau": float(getattr(args, "class_balance_tau", 1.0)),
            "server_momentum_beta": float(getattr(args, "server_momentum_beta", 0.0)),
            "round_lr_schedule": str(getattr(args, "round_lr_schedule", "linear")),
            "val_selection_ema": float(getattr(args, "val_selection_ema", 0.0)),
            "test_eval_every_round": int(getattr(args, "test_eval_every_round", 1)),
            "attn_implementation": str(getattr(args, "attn_implementation", "sdpa")),
            "smids_head_target_names": list(getattr(args, "smids_head_target_names", [])),
            "smids_tail_target_names": list(getattr(args, "smids_tail_target_names", [])),
            "sipakmorph_feature_columns": list(getattr(args, "morph_feature_columns", [])),
            "adaptive_early_stop_patience": int(getattr(args, "adaptive_early_stop_patience", 0)),
            "round_lr_scale_start": float(getattr(args, "round_lr_scale_start", 1.0)),
            "round_lr_scale_end": float(getattr(args, "round_lr_scale_end", 0.50)),
            "round_lr_decay_start_round": int(getattr(args, "round_lr_decay_start_round", 6)),
        },
        "server_visibility": visibility,
        "data_split": {
            "global_test_ratio": float(split_obj.get("global_test_ratio", 0.0)),
            "experiment_type": str(split_obj.get("experiment_type", "class_skew")),
            "domain_shift_enabled": bool(split_obj.get("client_domain_profiles")),
            "requested_domain_shift_level": str(split_obj.get("requested_domain_shift_level", "")),
            "domain_shift_severity": str(split_obj.get("domain_shift_severity", "")),
            "domain_shift_factor": str(split_obj.get("domain_shift_factor", "")),
            "global_test_domain_eval": str(split_obj.get("global_test_domain_eval", "")),
            "client_domain_profiles": split_obj.get("client_domain_profiles", {}),
        },
    }


def build_summary_payload(header, best_row, final_row, per_round_logs, test_at_best_val=None, last_test_row=None):
    """Assemble results.json: header, selection info and the round history."""
    timing = header.get("timing", {})
    visibility = header.get("server_visibility", {})
    data_split = header.get("data_split", {})
    best_val = best_row.get("global_val", {})
    # With --test-eval-every-round > 1 the final round may skip the
    # monitor-only test evaluation; fall back to the most recent one.
    final_test = final_row.get("global_test") or last_test_row or {}
    return {
        "experiment": header,
        "morphfl": header.get("morphfl"),
        "model_selection": header.get("model_selection"),
        "best_round": int(best_row["round"]),
        "best_val_macro_f1": float(best_val.get("macro_f1", 0.0)),
        "best_global_val": best_val,
        "test_at_best_val": test_at_best_val,
        "test_macro_f1_at_best_val": (
            float(test_at_best_val.get("macro_f1", 0.0)) if test_at_best_val else None
        ),
        "final_round_test": final_test,
        "final_round_test_macro_f1": float(final_test.get("macro_f1", 0.0)) if final_test else None,
        "final_round_val_macro_f1": float(final_row.get("global_val", {}).get("macro_f1", 0.0)),
        "final_round": final_row,
        "rounds": per_round_logs,
        "total_train_time_sec": timing.get("total_train_time_sec"),
        "avg_round_time_sec": timing.get("avg_round_time_sec"),
        "shared_scope": header.get("shared_scope"),
        "private_scope": header.get("private_scope"),
        "data_split": data_split,
        "requested_domain_shift_level": data_split.get("requested_domain_shift_level"),
        "domain_shift_severity": data_split.get("domain_shift_severity"),
        "domain_shift_factor": data_split.get("domain_shift_factor"),
        "global_test_domain_eval": data_split.get("global_test_domain_eval"),
        "client_domain_profiles": data_split.get("client_domain_profiles"),
        "shared_param_count": visibility.get("shared_param_count"),
        "private_param_count": visibility.get("private_param_count"),
        "shared_param_ratio": visibility.get("shared_param_ratio"),
        "shared_state_bytes": visibility.get("shared_state_bytes"),
        "per_round_total_bytes": visibility.get("per_round_total_bytes"),
        "total_communication_bytes": visibility.get("total_communication_bytes"),
    }
