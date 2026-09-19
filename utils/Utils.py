"""Seeding, JSON/CLI helpers and multiclass metrics."""


import argparse
import json
import math
import os
import random
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def set_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def compute_metrics_multiclass(
    preds: np.ndarray,
    targets: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(preds.max(), targets.max()) + 1)

    metrics: Dict[str, object] = {}
    metrics["acc"] = float(np.mean(preds == targets))
    metrics["macro_f1"] = float(f1_score(targets, preds, average="macro"))
    metrics["weighted_f1"] = float(f1_score(targets, preds, average="weighted"))

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    if class_names is None:
        resolved_class_names = [f"class{i}" for i in range(num_classes)]
    else:
        resolved_class_names = list(class_names)
        if len(resolved_class_names) < num_classes:
            resolved_class_names.extend(
                f"class{i}" for i in range(len(resolved_class_names), num_classes)
            )

    total = float(cm.sum())
    metrics["num_classes"] = int(num_classes)
    metrics["total_samples"] = int(total)
    metrics["class_names"] = resolved_class_names
    metrics["confusion_matrix"] = cm.astype(int).tolist()

    class_precisions, class_recalls, class_f1s, class_supports = [], [], [], []
    per_class: Dict[str, Dict[str, float]] = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_accuracy = (tp + tn) / total if total > 0 else 0.0
        support = tp + fn

        metrics[f"class{i}_accuracy"] = float(class_accuracy)
        metrics[f"class{i}_precision"] = float(precision)
        metrics[f"class{i}_recall"] = float(recall)
        metrics[f"class{i}_specificity"] = float(specificity)
        metrics[f"class{i}_f1"] = float(f1)
        metrics[f"class{i}_support"] = float(support)
        per_class[resolved_class_names[i]] = {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "accuracy": float(class_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "support": int(support),
        }

        class_precisions.append(float(precision))
        class_recalls.append(float(recall))
        class_f1s.append(float(f1))
        class_supports.append(float(support))

    support_sum = float(sum(class_supports))
    metrics["macro_precision"] = float(np.mean(class_precisions)) if class_precisions else 0.0
    metrics["macro_recall"] = float(np.mean(class_recalls)) if class_recalls else 0.0
    metrics["balanced_acc"] = float(np.mean(class_recalls)) if class_recalls else 0.0
    metrics["weighted_precision"] = (
        float(np.average(class_precisions, weights=class_supports)) if support_sum > 0 else 0.0
    )
    metrics["weighted_recall"] = (
        float(np.average(class_recalls, weights=class_supports)) if support_sum > 0 else 0.0
    )

    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(targets, probs, average="macro", multi_class="ovr")
            )
            for i in range(num_classes):
                y_true_i = (targets == i).astype(int)
                y_score_i = probs[:, i]
                if len(np.unique(y_true_i)) > 1:
                    auc_i = float(roc_auc_score(y_true_i, y_score_i))
                else:
                    auc_i = float("nan")
                metrics[f"class{i}_auc"] = auc_i
                per_class[resolved_class_names[i]]["auc"] = auc_i
        except ValueError:
            pass

    metrics["per_class"] = per_class
    return metrics


def _infer_num_classes_from_metrics(metric: Mapping[str, float]) -> int:
    class_ids = []
    for key in metric:
        if not key.startswith("class"):
            continue
        digits = []
        for ch in key[5:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            class_ids.append(int("".join(digits)))
    return (max(class_ids) + 1) if class_ids else 0


def summarize_multiclass_metric_dicts(
    metric_rows: Sequence[Mapping[str, float]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if not metric_rows:
        return {"num_folds": 0, "overall": {}, "per_class": {}}

    num_classes = _infer_num_classes_from_metrics(metric_rows[0])
    if class_names is None:
        class_names = [f"class{i}" for i in range(num_classes)]
    else:
        class_names = list(class_names)
        if len(class_names) < num_classes:
            class_names = class_names + [f"class{i}" for i in range(len(class_names), num_classes)]

    overall_keys = [
        "acc",
        "balanced_acc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_auc",
    ]
    per_class_metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "support",
    ]

    def _finite_values(values):
        out = []
        for value in values:
            if value is None:
                continue
            value = float(value)
            if not math.isnan(value):
                out.append(value)
        return out

    def _mean_std(values):
        values = _finite_values(values)
        if not values:
            return {"mean": float("nan"), "std": float("nan")}
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return {"mean": float(np.mean(values)), "std": std}

    summary: Dict[str, object] = {"num_folds": len(metric_rows), "overall": {}, "per_class": {}}
    for key in overall_keys:
        stats = _mean_std([row.get(key) for row in metric_rows if key in row])
        if not math.isnan(stats["mean"]):
            summary["overall"][key] = stats

    for class_idx in range(num_classes):
        class_key = class_names[class_idx]
        class_stats = {}
        for metric_key in per_class_metric_keys:
            flat_key = f"class{class_idx}_{metric_key}"
            stats = _mean_std([row.get(flat_key) for row in metric_rows if flat_key in row])
            if not math.isnan(stats["mean"]):
                class_stats[metric_key] = stats
        summary["per_class"][class_key] = class_stats

    return summary
