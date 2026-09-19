"""Five-fold membership-inference audit (loss/confidence based)."""


from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from audits.Representations import exp_forward
from utils.Federation import build_loader


def _sample_paths(paths: List[str], k: int, rng: np.random.RandomState) -> List[str]:
    if len(paths) <= k:
        return list(paths)
    return [paths[int(i)] for i in rng.choice(len(paths), size=k, replace=False)]


def _collect_loss_stats(model, loader, exp, use_amp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    losses, confidences, corrects, base_losses = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out, labels = exp_forward(exp, batch, use_amp=use_amp)
            probs = out["fusion_probs"].float()
            losses.append(F.cross_entropy(out["fusion_logits"].float(), labels, reduction="none").cpu().numpy())
            confidences.append(probs.gather(1, labels.unsqueeze(1)).squeeze(1).cpu().numpy())
            corrects.append((probs.argmax(dim=1) == labels).cpu().numpy().astype(np.int8))
            base_losses.append(F.cross_entropy(out["base_logits"].float(), labels, reduction="none").cpu().numpy())
    return (
        np.concatenate(losses) if losses else np.zeros(0),
        np.concatenate(confidences) if confidences else np.zeros(0),
        np.concatenate(corrects) if corrects else np.zeros(0, dtype=np.int8),
        np.concatenate(base_losses) if base_losses else np.zeros(0),
    )


def membership_inference(exp: Dict, max_train_samples: int = 1200) -> Dict:
    device = exp["device"]
    model = exp["global_model"]
    split_obj = exp["split_obj"]
    data_dir = exp["data_dir"]
    class_to_idx = exp["class_to_idx"]
    morph_feature_columns = exp["morph_feature_columns"]
    profiles = exp["client_domain_profiles"]

    member_paths: List[str] = []
    for client_paths in split_obj["clients"].values():
        member_paths.extend(client_paths)
    non_member_paths: List[str] = list(split_obj["global_test"])
    rng = np.random.RandomState(42)
    member_paths = _sample_paths(member_paths, min(max_train_samples, len(member_paths)), rng)
    non_member_paths = _sample_paths(non_member_paths, min(max_train_samples, len(non_member_paths)), rng)

    eval_bs = int(getattr(exp["args"], "eval_batch_size", 96))
    num_workers = int(getattr(exp["args"], "num_workers", 2))
    use_amp = bool(getattr(exp["args"], "amp_enabled", True)) and device.type == "cuda"
    ref_mean = tuple(float(v) for v in getattr(exp["args"], "stain_ref_mean", []) or [])
    ref_std = tuple(float(v) for v in getattr(exp["args"], "stain_ref_std", []) or [])
    stain_kwargs = {
        "stain_ref_mean": ref_mean or None,
        "stain_ref_std": ref_std or None,
    }

    member_parts = []
    quota = max(1, len(member_paths) // max(1, len(split_obj["clients"])))
    for client_name, client_paths in split_obj["clients"].items():
        client_sample = _sample_paths(client_paths, quota, rng)
        if not client_sample:
            continue
        _, loader = build_loader(
            data_dir,
            client_sample,
            batch_size=eval_bs,
            num_workers=num_workers,
            class_to_idx=class_to_idx,
            shuffle=False,
            domain_shift_profile=profiles.get(client_name),
            morph_feature_columns=morph_feature_columns,
            persistent_workers=False,
            **stain_kwargs,
        )
        member_parts.append(_collect_loss_stats(model, loader, exp, use_amp))
    _, non_member_loader = build_loader(
        data_dir,
        non_member_paths,
        batch_size=eval_bs,
        num_workers=num_workers,
        class_to_idx=class_to_idx,
        shuffle=False,
        domain_shift_profile=profiles.get(exp["ref_client"]),
        morph_feature_columns=morph_feature_columns,
        persistent_workers=False,
        **stain_kwargs,
    )

    m_loss = np.concatenate([p[0] for p in member_parts]) if member_parts else np.zeros(0)
    m_conf = np.concatenate([p[1] for p in member_parts]) if member_parts else np.zeros(0)
    m_corr = np.concatenate([p[2] for p in member_parts]) if member_parts else np.zeros(0, np.int8)
    m_base = np.concatenate([p[3] for p in member_parts]) if member_parts else np.zeros(0)
    n_loss, n_conf, n_corr, n_base = _collect_loss_stats(model, non_member_loader, exp, use_amp)

    def auc(member_scores, non_member_scores) -> float:
        if len(member_scores) == 0 or len(non_member_scores) == 0:
            return float("nan")
        y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        scores = np.concatenate([member_scores, non_member_scores])
        if len(np.unique(y)) < 2:
            return float("nan")
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y, scores))
        except Exception:
            return float("nan")

    results = {
        "n_member": int(len(m_loss)),
        "n_non_member": int(len(n_loss)),
        "mean_loss_member": float(m_loss.mean()) if len(m_loss) else float("nan"),
        "mean_loss_non_member": float(n_loss.mean()) if len(n_loss) else float("nan"),
        "mean_conf_member": float(m_conf.mean()) if len(m_conf) else float("nan"),
        "mean_conf_non_member": float(n_conf.mean()) if len(n_conf) else float("nan"),
        "correct_rate_member": float(m_corr.mean()) if len(m_corr) else float("nan"),
        "correct_rate_non_member": float(n_corr.mean()) if len(n_corr) else float("nan"),
    }
    results["attack_auc_loss_based"] = auc(-m_loss, -n_loss)
    results["attack_auc_confidence_based"] = auc(m_conf, n_conf)
    results["attack_auc_base_loss_based"] = auc(-m_base, -n_base)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        x_member = np.stack([m_loss, m_conf, m_corr.astype(np.float32), m_base, -m_loss], axis=1)
        x_non = np.stack([n_loss, n_conf, n_corr.astype(np.float32), n_base, -n_loss], axis=1)
        x = np.concatenate([x_member, x_non], axis=0)
        y = np.concatenate([np.ones(len(x_member)), np.zeros(len(x_non))])
        if len(np.unique(y)) >= 2 and len(x) >= 20:
            classifier = LogisticRegression(max_iter=3000)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(classifier, x, y, cv=cv, scoring="roc_auc")
            results["attack_auc_lr_cv5_mean"] = float(cv_scores.mean())
            results["attack_auc_lr_cv5_std"] = float(cv_scores.std())
    except Exception as exc:
        results["attack_auc_lr_error"] = str(exc)
    return results
