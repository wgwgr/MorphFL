"""Load trained checkpoints and rebuild the audited experiment bundle."""


import argparse
import os
from typing import Dict

import torch

from audits.Baselines import BaselineModel
from utils.Datasets import SIPAKMORPH_FEATURE_COLUMNS
from utils.Federation import (
    build_loader,
    build_model,
    load_split_json,
    resolve_data_dir,
    validate_grouped_partition,
)
from utils.BoundaryScope import (
    DATASET_CLASS_NAMES,
    get_shared_private_keys,
    load_state_subset,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dict_to_namespace(value):
    """Recursively convert nested dicts into an argparse.Namespace."""
    if isinstance(value, dict):
        namespace = argparse.Namespace()
        for key, sub_value in value.items():
            setattr(namespace, key, dict_to_namespace(sub_value))
        return namespace
    return value


def _resolve_path(path: str) -> str:
    """Resolve a possibly-relative path against repo/workspace roots."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in (REPO_ROOT, os.path.dirname(REPO_ROOT)):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return path


def load_checkpoint(ckpt_dir: str):
    """Load a run directory's checkpoint_last.pth (or a direct .pth path)."""
    resolved = _resolve_path(ckpt_dir)
    if os.path.isdir(resolved):
        resolved = os.path.join(resolved, "checkpoint_last.pth")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"No checkpoint at {resolved} (from {ckpt_dir})")
    ckpt = torch.load(resolved, map_location="cpu")
    args_saved = ckpt.get("args", {}) if isinstance(ckpt.get("args"), dict) else {}
    summary = ckpt.get("summary", {})
    return ckpt, args_saved, summary, resolved


def _build_test_loader(data_dir, split_obj, args, class_names, class_to_idx, is_smids, device,
                       eval_batch_size, num_workers, morph_feature_columns):
    """Build the shared global-test loader used by all audit attacks."""
    profiles = split_obj.get("client_domain_profiles", {})
    ref_client = next(iter(split_obj["clients"].keys()))
    ref_mean = tuple(float(v) for v in getattr(args, "stain_ref_mean", [0.0, 0.0, 0.0]))
    ref_std = tuple(float(v) for v in getattr(args, "stain_ref_std", [1.0, 1.0, 1.0]))
    if not is_smids and not morph_feature_columns:
        morph_feature_columns = list(SIPAKMORPH_FEATURE_COLUMNS)
    _, loader = build_loader(
        data_dir=data_dir,
        split_list=split_obj["global_test"],
        batch_size=eval_batch_size,
        num_workers=num_workers,
        class_to_idx=class_to_idx,
        shuffle=False,
        domain_shift_profile=profiles.get(ref_client),
        morph_feature_columns=morph_feature_columns,
        persistent_workers=False,
        stain_ref_mean=ref_mean if any(abs(v) > 1e-8 for v in ref_mean) else None,
        stain_ref_std=ref_std,
    )
    return loader, profiles, ref_client


def _build_morphfl_experiment(ckpt, args_saved, summary, ckpt_path, device, eval_batch_size, num_workers):
    """Rebuild the MorphFL client from a checkpoint for offline auditing."""
    split_obj = load_split_json(_resolve_path(args_saved["split_json"]))
    validate_grouped_partition(split_obj)
    data_dir = resolve_data_dir(split_obj)
    is_smids = bool(args_saved.get("use_hierarchical_head", False))
    class_names = list(args_saved.get("class_names", list(DATASET_CLASS_NAMES)))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    args = dict_to_namespace(args_saved)
    args.method = "morphfl"
    if not hasattr(args, "eval_batch_size"):
        args.eval_batch_size = eval_batch_size
    if not hasattr(args, "num_workers"):
        args.num_workers = num_workers
    if not hasattr(args, "amp_enabled"):
        args.amp_enabled = bool(args_saved.get("amp_enabled", True))

    model = build_model(args, device, data_dir)
    load_state_subset(model, ckpt["global_state_dict"])
    client_private_states = ckpt.get("client_private_states", {})
    if client_private_states:
        first_client = next(iter(client_private_states))
        load_state_subset(model, client_private_states[first_client])

    morph_feature_columns = list(args_saved.get("morph_feature_columns", []))
    test_loader, profiles, ref_client = _build_test_loader(
        data_dir, split_obj, args, class_names, class_to_idx, is_smids, device,
        eval_batch_size, num_workers, morph_feature_columns,
    )
    shared_keys, private_keys = get_shared_private_keys(ckpt["global_state_dict"], args)
    return {
        "ckpt": ckpt,
        "ckpt_path": ckpt_path,
        "args": args,
        "args_saved": args_saved,
        "summary": summary,
        "data_dir": data_dir,
        "split_obj": split_obj,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": int(args_saved.get("num_classes", len(class_names))),
        "is_smids": is_smids,
        "client_names": list(split_obj["clients"].keys()),
        "global_model": model,
        "shared_keys": shared_keys,
        "private_keys": private_keys,
        "global_state": ckpt["global_state_dict"],
        "client_private_states": client_private_states,
        "test_loader": test_loader,
        "device": device,
        "morph_feature_columns": morph_feature_columns,
        "client_domain_profiles": profiles,
        "ref_client": ref_client,
        "model_config": {
            "backbone": str(getattr(model.frontend, "resolved_name", args_saved.get("pretrained_model_name", ""))),
            "tuning_mode": str(getattr(model.frontend, "tuning_mode", "unknown")),
            "use_lora": bool(getattr(model.frontend, "use_lora", False)),
            "lora_r": int(args_saved.get("lora_r", 8)) if getattr(model.frontend, "use_lora", False) else 0,
            "lora_alpha": int(args_saved.get("lora_alpha", 16)),
            "lora_target_modules": list(getattr(model.frontend, "lora_target_modules", [])),
            "train_norm_layers": bool(args_saved.get("train_norm_layers", False)),
        },
        "variant": "morphfl",
    }


def _build_baseline_experiment(ckpt, args_saved, summary, ckpt_path, device, eval_batch_size, num_workers):
    """Rebuild a baseline model (fedavg/fedbn/fedprox/ditto/fedproto/local_only)."""
    split_obj = load_split_json(_resolve_path(args_saved["split_json"]))
    validate_grouped_partition(split_obj)
    data_dir = resolve_data_dir(split_obj)
    class_names = list(args_saved.get("class_names", list(DATASET_CLASS_NAMES)))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    is_smids = class_names == list(DATASET_CLASS_NAMES)
    method = args_saved.get("method", "fedavg")
    train_norm = bool(method == "fedbn")
    # Baselines fully fine-tune the backbone by default. Legacy frozen+LoRA
    # checkpoints are detected by the stored lora_r and loaded for auditing.
    full_finetune = bool(args_saved.get("full_finetune_backbone", True))
    lora_r = int(args_saved.get("lora_r", 8)) if not full_finetune else 0

    model = BaselineModel(
        pretrained_model_name=args_saved.get("pretrained_model_name", "dinov3-vitb16-pretrain-lvd1689m"),
        num_classes=int(args_saved.get("num_classes", len(class_names))),
        class_names=class_names,
        full_finetune=full_finetune,
        lora_r=lora_r,
        lora_alpha=int(args_saved.get("lora_alpha", 16)),
        train_norm_layers=train_norm,
    ).to(device)
    global_state = ckpt["global_state_dict"]
    private_states = ckpt.get("client_private_states", {}) or {}
    first_private = next(iter(private_states.values()), None)
    missing, unexpected = model.load_from_global_and_private(global_state, first_private)
    print(
        f"  BaselineModel [{method}] full_finetune={full_finetune} train_norm={train_norm} | "
        f"missing tail={missing[-3:]} unexpected tail={unexpected[-3:]}"
    )

    args = dict_to_namespace(args_saved)
    args.method = method
    args.eval_batch_size = eval_batch_size
    args.num_workers = num_workers
    args.amp_enabled = bool(args_saved.get("amp_enabled", True))

    morph_feature_columns = list(args_saved.get("morph_feature_columns", []))
    test_loader, profiles, ref_client = _build_test_loader(
        data_dir, split_obj, args, class_names, class_to_idx, is_smids, device,
        eval_batch_size, num_workers, morph_feature_columns,
    )

    named_params = [name for name, _ in model.named_parameters()]
    if method in ("fedavg", "fedprox"):
        shared_keys = named_params
        private_keys = []
    elif method == "ditto":
        private_keys = sorted({n for n in named_params if n.startswith("classifier_head.")})
        shared_keys = sorted(set(named_params) - set(private_keys))
    elif method in ("fedproto", "local_only"):
        shared_keys = []
        private_keys = named_params
    else:  # fedbn
        private_keys = sorted({
            n for n, p in model.named_parameters() if "norm" in n.lower() and p.requires_grad
        })
        shared_keys = sorted(set(named_params) - set(private_keys))

    prototypes = global_state.get("fedproto_prototypes", None)
    if prototypes is not None:
        prototypes = prototypes.to(device)
    return {
        "ckpt": ckpt,
        "ckpt_path": ckpt_path,
        "args": args,
        "args_saved": args_saved,
        "summary": summary,
        "data_dir": data_dir,
        "split_obj": split_obj,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": int(args_saved.get("num_classes", len(class_names))),
        "is_smids": is_smids,
        "client_names": list(split_obj["clients"].keys()),
        "global_model": model,
        "shared_keys": shared_keys,
        "private_keys": private_keys,
        "global_state": global_state,
        "client_private_states": ckpt.get("client_private_states", {}) or {},
        "test_loader": test_loader,
        "device": device,
        "morph_feature_columns": morph_feature_columns,
        "client_domain_profiles": profiles,
        "ref_client": ref_client,
        "model_config": {
            "backbone": str(args_saved.get("pretrained_model_name", "")),
            "tuning_mode": "full_finetune" if full_finetune else ("lora" if lora_r > 0 else "frozen"),
            "use_lora": bool(not full_finetune and lora_r > 0),
            "lora_r": int(lora_r),
            "lora_alpha": int(args_saved.get("lora_alpha", 16)),
            "lora_target_modules": ["q_proj", "v_proj"] if (not full_finetune and lora_r > 0) else [],
            "train_norm_layers": bool(train_norm),
        },
        "variant": "baseline",
        "fedproto_prototypes": prototypes,
    }


def prepare_experiment(
    ckpt_dir: str,
    device: torch.device,
    eval_batch_size: int = 96,
    num_workers: int = 2,
) -> Dict:
    """Dispatch a checkpoint to the right experiment builder by method."""
    ckpt, args_saved, summary, ckpt_path = load_checkpoint(ckpt_dir)
    method = args_saved.get("method", "morphfl")
    if method == "morphfl":
        builder = _build_morphfl_experiment
    elif method in ("fedavg", "fedbn", "fedprox", "ditto", "fedproto", "local_only"):
        builder = _build_baseline_experiment
    else:
        raise ValueError(f"Unsupported method in checkpoint: {method}")
    return builder(
        ckpt=ckpt,
        args_saved=args_saved,
        summary=summary,
        ckpt_path=ckpt_path,
        device=device,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
