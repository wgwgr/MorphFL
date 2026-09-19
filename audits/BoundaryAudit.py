"""Semantic probes and the per-group boundary certificate.

Two complementary questions are audited:

1. can image content be reconstructed from each shared parameter group via
   DLG gradient inversion (expected to fail);
2. can the declared geometric measurements be linearly recovered from the
   shared representations via five-fold ridge regression (expected to
   succeed -- the shared surface is supposed to carry morphology).
"""


import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from audits.GradientInversion import gradient_inversion_dlg
from audits.Representations import collect_representations
from utils.Utils import write_json


# (group name, input modality tag, parameter prefixes) as presented in the
# boundary certificate table.
BOUNDARY_GROUPS: List[Tuple[str, str, List[str]]] = [
    ("anchor", "scalar", ["mem.anchor_classifier.", "mem.scalar_norm."]),
    ("reliability_heads", "measurement", [
        "urc.measurement_head.", "urc.visual_head.", "urc.gate_bias",
    ]),
    ("compilation_mlps", "scalar+quality", ["mem.compiler_head.", "mem.compiler_tail."]),
    ("target_predictors", "scalar+quality", [
        "mem.head_morph_predictor.",
        "mem.tail_morph_predictor.",
        "mem.head_evidence_encoder.",
        "mem.tail_evidence_encoder.",
    ]),
    ("vsp_adapter", "visual", ["vsp.morph_adapter."]),
    ("vsp_concept_projector", "visual", ["vsp.concept_projector."]),
    ("context_encoders", "mixed", ["vsp.context_encoder.", "vsp.local_summary_encoder."]),
    ("compiled_classifier", "mixed", ["mem.compiled_classifier."]),
    ("decision_heads", "visual", [
        "vsp.ns_gate.", "vsp.abn_cls.", "vsp.base_classifier.",
    ]),
]


def ridge_attribute_r2(exp: Dict, max_samples: int = 512, n_folds: int = 5) -> Dict:
    """Five-fold out-of-fold ridge R^2 of measurements per representation."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    collected = collect_representations(exp, max_samples=max_samples)
    scalars = collected["scalar"]
    if scalars.numel() == 0 or scalars.dim() < 2:
        return {"note": "no scalars collected"}
    targets = scalars.numpy()
    n_samples, n_targets = targets.shape
    repr_keys = [
        "compiled_head_concept",
        "compiled_tail_concept",
        "base_concept",
        "global_context",
        "private_feature",
    ]
    results: Dict[str, Dict] = {
        "n_samples": int(n_samples),
        "n_scalars": int(n_targets),
        "cv_folds": n_folds,
    }
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for key in repr_keys:
        features = collected.get(key)
        if features is None or features.numel() == 0 or features.dim() < 2:
            continue
        x = features.numpy()
        per_scalar_r2 = np.zeros(n_targets)
        for col in range(n_targets):
            y = targets[:, col]
            if float(np.std(y)) < 1e-8:
                continue
            predictions = np.zeros(n_samples)
            for train_idx, test_idx in kf.split(x):
                scaler = StandardScaler().fit(x[train_idx])
                model = Ridge(alpha=1.0).fit(scaler.transform(x[train_idx]), y[train_idx])
                predictions[test_idx] = model.predict(scaler.transform(x[test_idx]))
            ss_res = float(np.sum((y - predictions) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
            per_scalar_r2[col] = 1.0 - ss_res / ss_tot
        results[key] = {
            "mean_r2": float(np.mean(per_scalar_r2)),
            "per_scalar_r2": [float(v) for v in per_scalar_r2],
        }
        print(f"      {key}: mean R2={results[key]['mean_r2']:.3f}")
    return results


def boundary_certificate(
    exp: Dict,
    output_dir: str,
    dlg_iterations: int = 600,
    dlg_restarts: int = 2,
    n_trials: int = 2,
    full_iterations: int = 2000,
    full_restarts: int = 3,
    group_filter: Optional[List[str]] = None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    candidate_groups = [g for g in BOUNDARY_GROUPS if not group_filter or g[0] in group_filter]
    shared_keys = list(exp.get("shared_keys", []))
    groups = [g for g in candidate_groups if any(k.startswith(tuple(g[2])) for k in shared_keys)]
    certificate: Dict[str, object] = {
        "groups": [{"group": g, "modality": modality, "prefixes": prefixes}
                   for g, modality, prefixes in groups]
    }

    per_group: Dict[str, Dict] = {}
    for group, _modality, prefixes in groups:
        print(f"    [cert] DLG group = {group}")
        result = gradient_inversion_dlg(
            exp,
            os.path.join(output_dir, f"dlg_{group}"),
            n_trials=n_trials,
            iterations=dlg_iterations,
            n_restarts=dlg_restarts,
            shared_key_filter=prefixes,
            setups=[("shared_params_only", True)],
        )
        per_group[group] = result.get("shared_params_only", {})
    certificate["dlg_per_group"] = per_group

    print("    [cert] DLG full-surface reference")
    full_reference = gradient_inversion_dlg(
        exp,
        os.path.join(output_dir, "dlg_full_shared"),
        n_trials=max(n_trials, 3),
        iterations=full_iterations,
        n_restarts=full_restarts,
        setups=[("shared_params_only", True)],
    )
    certificate["dlg_full_shared_reference"] = full_reference.get("shared_params_only", {})

    print("    [cert] ridge attribute R2")
    certificate["ridge_attribute_r2"] = ridge_attribute_r2(exp)

    write_json(os.path.join(output_dir, "boundary_certificate.json"), certificate)
    return certificate
