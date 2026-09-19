"""Federated training protocol (server broadcast, local updates, aggregation).

Each round the server broadcasts the shared parameters; each client jointly
optimizes the shared and local parameters on its local objective and uploads
only the shared subset, which is averaged with sample-size weights. The
module also implements the matched train/test shift protocol, per-round
global evaluation and checkpoint/result serialization.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.Datasets import (
    SIPAKMORPH_FEATURE_COLUMNS,
    infer_label_from_relpath,
    parent_image_key,
    SMIDSDataset,
    SIPaKMeDDataset,
    build_class_to_idx,
    collate_fn,
)
from utils.LocalTraining import train_local
from utils.Evaluation import (
    attach_shift_breakdown,
    build_result_header,
    build_summary_payload,
    evaluate_personalized_models,
    evaluate_personalized_models_shifted_concat,
    strip_metric_keys,
    summarize_timing,
    write_round_log_artifacts,
)
from utils.BoundaryScope import (
    CLASS_DISPLAY_NAMES,
    DATASET_CLASS_NAMES,
    average_states,
    get_shared_private_keys,
    get_trainable_state_keys,
    load_state_subset,
    snapshot_model_state,
    snapshot_state_subset,
)
from model.MorphFL import MorphFLModel
from utils.Utils import set_seed, write_json


MORPH_SAFE_AUG = transforms.Compose([
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.04, hue=0.015)],
        p=0.9,
    ),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.25)], p=0.2),
    transforms.RandomAutocontrast(p=0.1),
])

MORPH_SAFE_AUG_CONFIG = {
    "name": "morphology_safe_augmentation",
    "geometry_preserving": True,
    "ops": [
        "ColorJitter(brightness=0.12, contrast=0.12, saturation=0.04, hue=0.015), p=0.9",
        "GaussianBlur(kernel_size=3, sigma=0.1-0.6), p=0.2",
        "RandomAdjustSharpness(1.25), p=0.2",
        "RandomAutocontrast, p=0.1",
    ],
}


def should_adaptive_early_stop(per_round_logs: List[Mapping[str, object]], args) -> Tuple[bool, dict | None]:
    """Decide whether to stop training early on stalled validation progress.

    Returns ``(True, stop_info)`` once the current round exceeds the best
    validation round by ``adaptive_early_stop_patience`` rounds and the
    best metric leads the current one by at least ``min_delta``.
    """
    patience = max(int(getattr(args, "adaptive_early_stop_patience", 0)), 0)
    if patience <= 0 or not per_round_logs:
        return False, None
    current_round = int(per_round_logs[-1].get("round", len(per_round_logs)))
    if current_round < max(int(getattr(args, "adaptive_early_stop_min_round", 0)), 0):
        return False, None
    min_delta = float(getattr(args, "adaptive_early_stop_min_delta", 1e-4))
    best_row = max(per_round_logs, key=lambda row: float(row.get("selection_metric", 0.0)))
    best_round = int(best_row.get("round", 1))
    if current_round <= best_round:
        return False, None
    if current_round - best_round < patience:
        return False, None
    current_metric = float(per_round_logs[-1].get("selection_metric", 0.0))
    best_metric = float(best_row.get("selection_metric", 0.0))
    if (best_metric - current_metric) < min_delta:
        return False, None
    return True, {
        "enabled": True,
        "reason": "no_improvement_after_patience",
        "current_round": current_round,
        "best_round": best_round,
        "rounds_since_best": current_round - best_round,
        "patience": patience,
        "best_metric": best_metric,
        "current_metric": current_metric,
    }


def _resolve_split_class_metadata(split_obj: Mapping[str, object]) -> Tuple[List[str], Dict[str, int], bool]:
    """Read class names from the split JSON, falling back to SMIDS defaults."""
    class_names = split_obj.get("class_names")
    if isinstance(class_names, list) and class_names:
        resolved_names = [str(name) for name in class_names]
    else:
        resolved_names = list(DATASET_CLASS_NAMES)
    class_to_idx = build_class_to_idx(resolved_names)
    return resolved_names, class_to_idx, resolved_names == list(DATASET_CLASS_NAMES)


def _safe_command_output(command: List[str], cwd: str | None = None) -> str | None:
    """Run a read-only command and return stripped stdout, or None on failure."""
    try:
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def collect_runtime_environment(device: torch.device) -> Dict[str, object]:
    """Capture the software/hardware environment of this run for provenance."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    environment = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "device": str(device),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "working_directory": repo_root,
        "git_commit": _safe_command_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_branch": _safe_command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "git_is_dirty": None,
    }
    dirty = _safe_command_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root)
    if dirty is not None:
        environment["git_is_dirty"] = bool(dirty)
    if torch.cuda.is_available():
        environment["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if device.type == "cuda" and device.index is not None:
            environment["current_gpu_index"] = int(device.index)
            environment["current_gpu_name"] = torch.cuda.get_device_name(device.index)
    return environment


def _normalize_class_counts(raw_counts: Mapping[str, object]) -> Dict[str, int]:
    """Map raw per-class counts (folder or display keys) onto display names."""
    normalized = {}
    for idx, dataset_name in enumerate(DATASET_CLASS_NAMES):
        display = CLASS_DISPLAY_NAMES[idx] if idx < len(CLASS_DISPLAY_NAMES) else dataset_name
        normalized[display] = int(raw_counts.get(dataset_name, raw_counts.get(display, 0)))
    return normalized


def _summarize_split_statistics(split_obj: Mapping[str, object]) -> Dict[str, object]:
    """Summarize partition sizes and class distributions for provenance."""
    stats = split_obj.get("stats", {}) if isinstance(split_obj, Mapping) else {}
    client_stats = stats.get("clients", {}) if isinstance(stats, Mapping) else {}
    size_stats = stats.get("sizes", {}) if isinstance(stats, Mapping) else {}
    return {
        "global_test_size": int(size_stats.get("global_test", len(split_obj.get("global_test", [])))),
        "global_val_size": int(size_stats.get("global_val", len(split_obj.get("global_val", [])))),
        "global_test_class_counts": _normalize_class_counts(stats.get("global_test", {})),
        "global_val_class_counts": _normalize_class_counts(stats.get("global_val", {})),
        "client_sizes": {
            name: int(size_stats.get(name, len(split_obj.get("clients", {}).get(name, []))))
            for name in split_obj.get("clients", {})
        },
        "client_class_counts": {
            name: _normalize_class_counts(client_stats.get(name, {}))
            for name in split_obj.get("clients", {})
        },
    }


def load_split_json(path: str) -> Dict[str, object]:
    """Read a split JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_dir(split_obj: Mapping[str, object]) -> str:
    """Resolve the dataset root recorded in a split JSON.

    Split JSONs generated under a previous repository layout may record a
    data_dir that no longer exists (e.g. ``../morphbasis_fl_noCCMD/data/SMIDS``).
    In that case fall back to the bundled ``<repo_root>/data/<basename>``
    directory so training keeps working after the repository was relocated.
    """
    data_dir = str(split_obj.get("data_dir", "") or "")
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    basename = os.path.basename(os.path.normpath(data_dir)) if data_dir else ""
    if not basename:
        raise FileNotFoundError(
            "Split JSON has no usable 'data_dir'; cannot resolve the dataset root."
        )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fallback = os.path.join(repo_root, "data", basename)
    if os.path.isdir(fallback):
        print(
            f"[Warning] Split data_dir '{data_dir}' does not exist; "
            f"falling back to '{fallback}'.",
            flush=True,
        )
        return fallback
    raise FileNotFoundError(
        f"Dataset directory not found: '{data_dir}' (fallback '{fallback}' also missing). "
        "Regenerate the split JSON or point data_dir at the dataset root."
    )


def validate_grouped_partition(split_obj: Mapping[str, object]) -> None:
    """Reject splits where one source image spans two partitions.

    SIPaKMeD crops are grouped by the parent cluster image, so the parent
    groups of the global test and validation sets and every client must be
    pairwise disjoint. Independent-image datasets (SMIDS) satisfy this
    trivially because each relpath is its own group. A global validation
    partition is mandatory.
    """
    if "global_val" not in split_obj:
        raise ValueError(
            "Split JSON is missing 'global_val'. Regenerate the split with "
            "scripts/make_splits.py."
        )
    groups = {
        "test": {parent_image_key(p) for p in split_obj.get("global_test", [])},
        "validation": {parent_image_key(p) for p in split_obj.get("global_val", [])},
    }
    if not groups["test"] or not groups["validation"]:
        raise ValueError("Both global_test and global_val must be non-empty.")
    groups.update({
        f"client:{name}": {parent_image_key(p) for p in paths}
        for name, paths in split_obj.get("clients", {}).items()
    })
    names = list(groups)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            overlap = groups[name_a] & groups[name_b]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"Data leakage: {name_a} and {name_b} share "
                    f"{len(overlap)} split unit(s), e.g. {sample}. Regenerate "
                    "the split with scripts/make_splits.py."
                )
    all_paths = (
        list(split_obj.get("global_test", []))
        + list(split_obj.get("global_val", []))
        + [p for paths in split_obj.get("clients", {}).values() for p in paths]
    )
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Data leakage: a sample appears in more than one partition.")

    # If an explicit pooled training list exists it must be exactly the
    # disjoint union of the client partitions.
    pooled_train = split_obj.get("train")
    if pooled_train is not None:
        client_union = sorted(p for paths in split_obj.get("clients", {}).values() for p in paths)
        if sorted(pooled_train) != client_union:
            raise ValueError(
                "'train' must be the exact union of the client partitions."
            )


def load_stain_reference(use_smids: bool):
    """Training-distribution RGB reference for evaluation stain normalization."""
    ref_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "assets", "stain_reference.json")
    )
    if not os.path.isfile(ref_path):
        return None, None
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            references = json.load(f)
        entry = (references or {}).get("smids" if use_smids else "sipak")
        if not entry:
            return None, None
        return tuple(float(v) for v in entry["mean"]), tuple(float(v) for v in entry["std"])
    except Exception:
        return None, None


def _seed_worker(worker_id: int) -> None:
    """Seed numpy/random per DataLoader worker from the worker's torch seed.

    Keeps per-worker domain-shift noise (which samples the global numpy RNG in
    utils/DomainShift.py) independent across workers and reproducible across
    runs instead of identical across workers.
    """
    import random

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    data_dir,
    split_list,
    batch_size,
    num_workers,
    class_to_idx,
    transform=None,
    morph_stats=None,
    shuffle=False,
    domain_shift_profile=None,
    morph_feature_columns=None,
    persistent_workers=None,
    stain_ref_mean=None,
    stain_ref_std=None,
    seed=None,
):
    """Build a dataset+DataLoader pair for one partition.

    ``seed`` enables a deterministic shuffle generator and per-worker
    numpy/random seeding (needed because domain-shift noise samples the
    global numpy RNG); ``persistent_workers`` defaults to True whenever
    worker processes are used.
    """
    use_smids_dataset = list(class_to_idx.keys()) == list(DATASET_CLASS_NAMES)
    if use_smids_dataset:
        dataset = SMIDSDataset(
            data_dir,
            split_list=split_list,
            transform=transform,
            morph_stats=morph_stats,
            domain_shift_profile=domain_shift_profile,
            stain_ref_mean=stain_ref_mean,
            stain_ref_std=stain_ref_std,
        )
    else:
        dataset = SIPaKMeDDataset(
            data_dir,
            split_list=split_list,
            class_to_idx=class_to_idx,
            transform=transform,
            morph_feature_columns=morph_feature_columns,
            morph_stats=morph_stats,
            domain_shift_profile=domain_shift_profile,
            stain_ref_mean=stain_ref_mean,
            stain_ref_std=stain_ref_std,
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = _seed_worker
    if persistent_workers is None:
        persistent_workers = int(num_workers) > 0
    if int(num_workers) > 0 and persistent_workers:
        loader_kwargs["persistent_workers"] = True
    return dataset, DataLoader(dataset, **loader_kwargs)


def build_model(args, device, data_dir: str, domain_shift_profile=None):
    """Instantiate the MorphFL client model from CLI arguments."""
    return MorphFLModel(
        data_dir=data_dir,
        pretrained_model_name=args.pretrained_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        train_norm_layers=bool(getattr(args, "train_norm_layers", False)),
        domain_shift_profile=domain_shift_profile,
        attn_implementation=str(getattr(args, "attn_implementation", "sdpa")),
        num_classes=int(getattr(args, "num_classes", len(DATASET_CLASS_NAMES))),
        class_names=list(getattr(args, "class_names", DATASET_CLASS_NAMES)),
        use_hierarchical_head=bool(getattr(args, "use_hierarchical_head", True)),
        scalar_feat_dim=int(getattr(args, "scalar_feat_dim", 2)),
        concept_dim=int(getattr(args, "concept_dim", 64)),
        morph_adapter_rank=int(getattr(args, "morph_adapter_rank", 16)),
        morph_adapter_alpha=float(getattr(args, "morph_adapter_alpha", 16.0)),
        adapter_dropout=float(getattr(args, "adapter_dropout", 0.1)),
        hidden_dim=int(getattr(args, "hidden_dim", 256)),
        share_global_context=bool(getattr(args, "share_global_context", True)),
        style_residual_scale=float(getattr(args, "style_residual_scale", 0.12)),
        style_logit_scale=float(getattr(args, "style_logit_scale", 0.05)),
        style_delta_max_norm=float(getattr(args, "style_delta_max_norm", 0.60)),
        compiler_input_dim=int(getattr(args, "compiler_input_dim", 6)),
        compiler_hidden_dim=int(getattr(args, "compiler_hidden_dim", 128)),
        content_conflict_decay=float(getattr(args, "content_conflict_decay", 0.30)),
        content_conflict_anchor_strength=float(getattr(args, "content_conflict_anchor_strength", 0.85)),
        compiler_conflict_decay=float(getattr(args, "compiler_conflict_decay", 0.75)),
        compiler_conflict_anchor_strength=float(getattr(args, "compiler_conflict_anchor_strength", 0.85)),
        anchor_floor=float(getattr(args, "anchor_floor", 0.0)),
        anchor_scale=float(getattr(args, "anchor_scale", 0.7)),
        anchor_conflict_boost=float(getattr(args, "anchor_conflict_boost", 0.80)),
        anchor_max=float(getattr(args, "anchor_max", 0.95)),
        anchor_takeover_mid=float(getattr(args, "anchor_takeover_mid", 0.45)),
        anchor_takeover_width=float(getattr(args, "anchor_takeover_width", 0.10)),
        rho_enabled=bool(getattr(args, "rho_enabled", False)),
        rho_warmup_rounds=int(getattr(args, "rho_warmup_rounds", 4)),
        rho_hidden_dim=int(getattr(args, "rho_hidden_dim", 32)),
    ).to(device)


def run_experiment(args):
    """Run the full federated training protocol and serialize all artifacts.

    Each round: restore the global shared state plus the client private
    state, run the local updates, aggregate the shared subset with
    sample-size weights (optionally with server momentum), then evaluate
    the personalized models on the global validation split (and, when
    ``--test-eval-every-round`` allows, on the test split for monitoring).
    The best-validation checkpoint is restored and evaluated once on the
    test split at the end; all artifacts are written to ``args.output_dir``.
    """
    set_seed(args.seed, False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_start_perf = time.perf_counter()
    split_obj = load_split_json(args.split_json)
    validate_grouped_partition(split_obj)
    data_dir = resolve_data_dir(split_obj)
    args.data_dir = data_dir
    class_names, class_to_idx, use_hierarchical_head = _resolve_split_class_metadata(split_obj)
    args.class_names = list(class_names)
    args.num_classes = int(len(class_names))
    args.use_hierarchical_head = bool(use_hierarchical_head)
    stain_ref_mean, stain_ref_std = load_stain_reference(bool(use_hierarchical_head))
    if stain_ref_mean is None or stain_ref_std is None:
        raise ValueError("Evaluation stain normalization requires data/assets/stain_reference.json.")

    if bool(args.use_hierarchical_head):
        args.scalar_feat_dim = 2
        args.morph_feature_columns = []
        args.smids_head_target_names = list(MorphFLModel.SMIDS_HEAD_TARGET_NAMES)
        args.smids_tail_target_names = list(MorphFLModel.SMIDS_TAIL_TARGET_NAMES)
    else:
        args.morph_feature_columns = list(SIPAKMORPH_FEATURE_COLUMNS)
        args.scalar_feat_dim = int(len(args.morph_feature_columns))
        args.smids_head_target_names = []
        args.smids_tail_target_names = []

    os.makedirs(args.output_dir, exist_ok=True)
    runtime_environment = collect_runtime_environment(device)
    split_statistics = _summarize_split_statistics(split_obj)
    saved_args = dict(vars(args))
    saved_args.update({
        "data_dir": data_dir,
        "class_names": list(class_names),
        "num_classes": int(len(class_names)),
        "use_hierarchical_head": bool(use_hierarchical_head),
        "train_augmentation": MORPH_SAFE_AUG_CONFIG,
        "runtime_environment": runtime_environment,
        "split_statistics": split_statistics,
        "created_at": run_started_at,
        "stain_ref_mean": list(stain_ref_mean),
        "stain_ref_std": list(stain_ref_std),
    })
    write_json(os.path.join(args.output_dir, "args.json"), saved_args)

    client_relpaths = split_obj["clients"]
    global_test_domain_eval = str(split_obj.get("global_test_domain_eval", "clean") or "clean")
    client_domain_profiles = split_obj.get("client_domain_profiles", {})

    global_model = build_model(args, device, data_dir)
    global_state = snapshot_model_state(global_model)
    shared_keys, private_keys = get_shared_private_keys(global_state, args)
    private_train_keys = get_trainable_state_keys(global_model, private_keys)
    global_private_train_state = snapshot_state_subset(global_model, private_train_keys)

    client_private_states: Dict[str, Dict[str, torch.Tensor]] = {}
    client_train_counts = {name: len(relpaths) for name, relpaths in client_relpaths.items()}
    per_round_logs: List[Mapping[str, object]] = []
    client_morph_stats = {}

    # Client-local label statistics (MBP-safe: never leaves the client).
    client_class_priors: Dict[str, torch.Tensor] = {}
    for client_name, relpaths in client_relpaths.items():
        counts = np.zeros(int(args.num_classes), dtype=np.float64)
        for rel in relpaths:
            try:
                counts[class_to_idx[infer_label_from_relpath(rel, class_to_idx)]] += 1.0
            except KeyError:
                continue
        total = float(counts.sum())
        if total > 0:
            prior = counts / total
        else:
            prior = np.full(int(args.num_classes), 1.0 / max(int(args.num_classes), 1))
        client_class_priors[client_name] = torch.tensor(prior, dtype=torch.float32)

    client_train_loaders = {}
    for client_index, (client_name, relpaths) in enumerate(client_relpaths.items()):
        train_ds, train_loader = build_loader(
            data_dir,
            relpaths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_to_idx=class_to_idx,
            transform=MORPH_SAFE_AUG,
            morph_stats=None,
            shuffle=True,
            domain_shift_profile=client_domain_profiles.get(client_name),
            morph_feature_columns=getattr(args, "morph_feature_columns", []),
            persistent_workers=True,
            seed=int(args.seed) + client_index,
        )
        client_train_loaders[client_name] = train_loader
        client_morph_stats[client_name] = train_ds.morph_stats

    eval_num_workers = int(getattr(args, "eval_num_workers", -1))
    if eval_num_workers < 0:
        eval_num_workers = int(args.num_workers)
    eval_loaders = {}
    for client_index, client_name in enumerate(client_relpaths):
        _, test_loader = build_loader(
            data_dir,
            split_obj["global_test"],
            batch_size=args.eval_batch_size,
            num_workers=eval_num_workers,
            class_to_idx=class_to_idx,
            shuffle=False,
            morph_stats=client_morph_stats.get(client_name),
            domain_shift_profile=client_domain_profiles.get(client_name),
            morph_feature_columns=getattr(args, "morph_feature_columns", []),
            persistent_workers=True,
            stain_ref_mean=stain_ref_mean,
            stain_ref_std=stain_ref_std,
            seed=int(args.seed) + 10000 + client_index,
        )
        eval_loaders[client_name] = test_loader

    global_val_paths = split_obj.get("global_val", [])
    if not global_val_paths:
        raise ValueError(
            "Split JSON is missing 'global_val'. Regenerate the split with "
            "scripts/make_splits.py so the model can be selected on validation "
            "and evaluated once on test."
        )
    val_loaders = {}
    for client_index, client_name in enumerate(client_relpaths):
        _, val_loader = build_loader(
            data_dir,
            global_val_paths,
            batch_size=args.eval_batch_size,
            num_workers=eval_num_workers,
            class_to_idx=class_to_idx,
            shuffle=False,
            morph_stats=client_morph_stats.get(client_name),
            domain_shift_profile=client_domain_profiles.get(client_name),
            morph_feature_columns=getattr(args, "morph_feature_columns", []),
            persistent_workers=True,
            stain_ref_mean=stain_ref_mean,
            stain_ref_std=stain_ref_std,
            seed=int(args.seed) + 20000 + client_index,
        )
        val_loaders[client_name] = val_loader

    client_models = {
        name: build_model(args, device, data_dir, domain_shift_profile=client_domain_profiles.get(name))
        for name in client_relpaths
    }

    def restore_state(client_name: str) -> Dict[str, torch.Tensor]:
        """Compose one client's state: global shared subset + private subset."""
        restore = {key: global_state[key] for key in shared_keys if key in global_state}
        if client_name in client_private_states:
            restore.update(client_private_states[client_name])
        else:
            restore.update(global_private_train_state)
        return restore

    best_metric_so_far = -1.0
    best_checkpoint_payload = None
    selection_ema: float | None = None
    val_selection_ema = float(getattr(args, "val_selection_ema", 0.0))
    use_val_ema = 0.0 < val_selection_ema < 1.0
    last_test_log: Dict[str, object] | None = None

    for round_idx in range(args.rounds):
        round_start = time.perf_counter()
        shared_states = []
        round_log = {"round": round_idx + 1, "clients": {}}
        args._current_round = round_idx + 1
        train_time_sum = 0.0

        for client_name, relpaths in client_relpaths.items():
            client_model = client_models[client_name]
            load_state_subset(client_model, restore_state(client_name))

            local_stats = train_local(
                client_model,
                client_train_loaders[client_name],
                args,
                device,
                local_class_prior=client_class_priors.get(client_name),
            )
            train_time_sum += float(local_stats.get("train_time_sec", 0.0))
            shared_states.append(snapshot_state_subset(client_model, shared_keys))
            if private_train_keys:
                client_private_states[client_name] = snapshot_state_subset(client_model, private_train_keys)

            client_log = {
                "train_size": len(relpaths),
                "train_loss": local_stats["loss"],
                "train_acc": local_stats["acc"],
            }
            for key, value in local_stats.items():
                if key not in {"loss", "acc"}:
                    client_log[key] = value
            round_log["clients"][client_name] = client_log

        aggregation_start = time.perf_counter()
        client_names = list(client_relpaths.keys())
        averaged = average_states(shared_states, [client_train_counts[name] for name in client_names])
        round_log["aggregation"] = {
            "strategy": "sample_size",
            "per_client": {
                name: {"sample_weight": float(client_train_counts[name])} for name in client_names
            },
        }
        aggregation_time = float(time.perf_counter() - aggregation_start)
        server_momentum_beta = float(getattr(args, "server_momentum_beta", 0.0))
        if 0.0 < server_momentum_beta < 1.0 and round_idx > 0:
            # Server-side momentum: interpolate the new aggregate with the
            # previous global shared state.
            for key, tensor in averaged.items():
                previous = global_state.get(key)
                if previous is not None and torch.is_floating_point(tensor):
                    averaged[key] = (
                        (1.0 - server_momentum_beta) * previous.float()
                        + server_momentum_beta * tensor.float()
                    ).to(dtype=tensor.dtype)
        for key, tensor in averaged.items():
            global_state[key] = tensor.clone()
        round_log["aggregation"]["server_momentum_beta"] = (
            server_momentum_beta if 0.0 < server_momentum_beta < 1.0 and round_idx > 0 else 0.0
        )

        for client_name in client_relpaths:
            load_state_subset(client_models[client_name], restore_state(client_name))

        val_start = time.perf_counter()
        global_val = evaluate_personalized_models(
            client_models, val_loaders, device, args=args
        )
        round_log["global_val"] = attach_shift_breakdown(global_val, client_domain_profiles)
        val_eval_time = float(time.perf_counter() - val_start)
        test_start = time.perf_counter()
        test_eval_every = max(int(getattr(args, "test_eval_every_round", 1)), 0)
        run_mid_test = test_eval_every > 0 and ((round_idx + 1) % test_eval_every == 0)
        if run_mid_test:
            if global_test_domain_eval == "shifted_global_concat":
                global_test = evaluate_personalized_models_shifted_concat(
                    client_models, eval_loaders, device, args=args
                )
            else:
                global_test = evaluate_personalized_models(
                    client_models, eval_loaders, device, args=args
                )
            round_log["global_test"] = attach_shift_breakdown(global_test, client_domain_profiles)
            round_log["global_test"] = strip_metric_keys(round_log["global_test"], {"macro_auc"})
            last_test_log = round_log["global_test"]
        else:
            global_test = None
        test_eval_time = float(time.perf_counter() - test_start)
        eval_time = val_eval_time + test_eval_time

        round_log["timing"] = {
            "train_time_sec_sum": float(train_time_sum),
            "train_time_sec_mean": float(train_time_sum / max(len(client_relpaths), 1)),
            "aggregation_time_sec": aggregation_time,
            "global_eval_time_sec": eval_time,
            "val_eval_time_sec": val_eval_time,
            "test_eval_time_sec": test_eval_time,
            "round_time_sec": float(time.perf_counter() - round_start),
        }
        round_log["global_val"] = strip_metric_keys(round_log["global_val"], {"macro_auc"})
        raw_val_metric = float(round_log["global_val"]["macro_f1"])
        if use_val_ema:
            # Smooth the selection metric so noisy val rounds do not move the
            # best-round choice; early stopping reads the same value.
            selection_ema = (
                raw_val_metric if selection_ema is None
                else val_selection_ema * selection_ema + (1.0 - val_selection_ema) * raw_val_metric
            )
            round_log["selection_metric"] = float(selection_ema)
            round_log["selection_metric_raw"] = raw_val_metric
        else:
            round_log["selection_metric"] = raw_val_metric
        per_round_logs.append(round_log)

        if best_checkpoint_payload is None or round_log["selection_metric"] > best_metric_so_far:
            best_metric_so_far = round_log["selection_metric"]
            best_checkpoint_payload = {
                "global_state_dict": {
                    k: (v.clone() if hasattr(v, "clone") else v) for k, v in global_state.items()
                },
                "client_private_states": {
                    name: {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state.items()}
                    for name, state in client_private_states.items()
                },
            }

        monitor_line = (
            f"[Round {round_idx + 1:02d}/{args.rounds}] "
            f"Val Macro-F1={round_log['global_val']['macro_f1']:.4f}"
        )
        if global_test is not None:
            monitor_line += f" | Test Macro-F1={global_test['macro_f1']:.4f} (monitor-only)"
        print(monitor_line, flush=True)

        should_stop, stop_info = should_adaptive_early_stop(per_round_logs, args)
        if should_stop:
            per_round_logs[-1]["adaptive_early_stop"] = stop_info
            print(
                f"[Adaptive Stop] round={stop_info['current_round']:02d} "
                f"best_round={stop_info['best_round']} "
                f"best_macro_f1={stop_info['best_metric']:.4f}",
                flush=True,
            )
            break

    best_row = max(per_round_logs, key=lambda row: row["selection_metric"])
    total_train_time = float(time.perf_counter() - run_start_perf)

    # Keep the final-round state for checkpoint_last.pth before restoring the
    # best-validation checkpoint for the single final test evaluation, so the
    # reported test score never participates in model selection.
    best_val_macro_f1 = float(best_row["selection_metric"])
    if best_checkpoint_payload is None:
        raise RuntimeError("No best checkpoint was recorded.")
    last_round_global_state = dict(global_state)
    last_round_private_states = {
        name: dict(state) for name, state in client_private_states.items()
    }
    for key, tensor in best_checkpoint_payload["global_state_dict"].items():
        global_state[key] = tensor.clone() if hasattr(tensor, "clone") else tensor
    client_private_states = {
        name: {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state.items()}
        for name, state in best_checkpoint_payload["client_private_states"].items()
    }
    for client_name in client_relpaths:
        load_state_subset(client_models[client_name], restore_state(client_name))

    test_eval_start = time.perf_counter()
    test_at_best_val = evaluate_personalized_models(client_models, eval_loaders, device, args=args)
    test_at_best_val = strip_metric_keys(
        attach_shift_breakdown(test_at_best_val, client_domain_profiles), {"macro_auc"}
    )
    test_eval_time = float(time.perf_counter() - test_eval_start)
    print(
        f"[Best-val round {int(best_row['round'])}] Val Macro-F1={best_val_macro_f1:.4f} "
        f"-> Test Macro-F1={test_at_best_val['macro_f1']:.4f} "
        f"Acc={test_at_best_val['acc']:.4f}",
        flush=True,
    )

    finished_at = datetime.now().isoformat(timespec="seconds")
    timing = summarize_timing(per_round_logs, total_train_time, run_started_at, finished_at)
    timing["test_eval_at_best_val_time_sec"] = test_eval_time
    artifacts = write_round_log_artifacts(args.output_dir, per_round_logs)

    header = build_result_header(args, split_obj, global_state, shared_keys, private_keys)
    header["timing"] = timing
    header["runtime_environment"] = runtime_environment
    header["data_split"].update(split_statistics)
    header["artifacts"] = artifacts
    header["model_selection"] = {
        "metric": "macro_f1_ema" if use_val_ema else "macro_f1",
        "partition": "global_val",
        "test_partition": "global_test",
        "best_round": int(best_row["round"]),
        "best_val_macro_f1": best_val_macro_f1,
    }

    summary = build_summary_payload(
        header, best_row, per_round_logs[-1], per_round_logs, test_at_best_val,
        last_test_row=last_test_log,
    )
    write_json(os.path.join(args.output_dir, "results.json"), summary)

    if bool(getattr(args, "save_checkpoints", True)):
        torch.save(
            {
                "global_state_dict": last_round_global_state,
                "client_private_states": last_round_private_states,
                "summary": summary,
                "args": saved_args,
                "checkpoint_round": int(per_round_logs[-1]["round"]),
            },
            os.path.join(args.output_dir, "checkpoint_last.pth"),
        )
    best_checkpoint_payload["summary"] = summary
    best_checkpoint_payload["args"] = saved_args
    best_checkpoint_payload["best_round"] = int(best_row["round"])
    best_checkpoint_payload["best_val_macro_f1"] = best_val_macro_f1
    if bool(getattr(args, "save_checkpoints", True)):
        torch.save(best_checkpoint_payload, os.path.join(args.output_dir, "checkpoint_best.pth"))
        suffix = "saved to checkpoint_best.pth"
    else:
        suffix = "(checkpoint saving disabled)"
    print(f"Best-val model {suffix}", flush=True)
    print(f"Saved results to: {os.path.join(args.output_dir, 'results.json')}", flush=True)
    return summary
