"""Shared/private boundary declaration, state snapshots and aggregation.

Membership of the shared set ``w_s`` is a static, auditable prefix list:

- default boundary: the VSP, MEM and URC parameters;
- narrow ablation: the scalar normalization and anchor classifier only;
- wide ablation: the default boundary plus the private visual frontend.

All functions here are deterministic and free of randomness.
"""

from typing import Dict, Iterable, List, Mapping, Tuple

import torch

from utils.Datasets import CLASS_TO_IDX


BACKBONE_CANONICAL_PREFIX = "frontend.backbone."
BACKBONE_ALIAS_PREFIX = "backbone."

DATASET_CLASS_NAMES = [name for name, _ in sorted(CLASS_TO_IDX.items(), key=lambda item: item[1])]
CLASS_DISPLAY_NAMES = ["Normal", "Abnormal", "Non-Sperm"]


def is_redundant_backbone_alias_key(key: str) -> bool:
    return key.startswith(BACKBONE_ALIAS_PREFIX)


def mirror_backbone_key(key: str) -> str | None:
    if key.startswith(BACKBONE_CANONICAL_PREFIX):
        return BACKBONE_ALIAS_PREFIX + key[len(BACKBONE_CANONICAL_PREFIX):]
    if key.startswith(BACKBONE_ALIAS_PREFIX):
        return BACKBONE_CANONICAL_PREFIX + key[len(BACKBONE_ALIAS_PREFIX):]
    return None


def _count_state_numel(state: Mapping[str, torch.Tensor], keys: Iterable[str]) -> int:
    return sum(int(state[key].numel()) for key in keys if state.get(key) is not None)


def _count_state_bytes(state: Mapping[str, torch.Tensor], keys: Iterable[str]) -> int:
    return sum(
        int(state[key].numel()) * int(state[key].element_size())
        for key in keys
        if state.get(key) is not None
    )


def summarize_server_visibility(
    global_state: Mapping[str, torch.Tensor],
    shared_keys: Iterable[str],
    private_keys: Iterable[str],
    rounds: int,
    num_clients: int,
) -> Dict[str, object]:
    """Parameter counts and per-round traffic across the declared boundary."""
    shared_param_count = _count_state_numel(global_state, shared_keys)
    private_param_count = _count_state_numel(global_state, private_keys)
    total_param_count = shared_param_count + private_param_count
    shared_param_ratio = (
        float(shared_param_count / total_param_count) if total_param_count > 0 else 0.0
    )

    shared_bytes = _count_state_bytes(global_state, shared_keys)
    private_bytes = _count_state_bytes(global_state, private_keys)
    total_bytes = shared_bytes + private_bytes

    per_round_upload = int(shared_bytes * num_clients)
    per_round_download = int(shared_bytes * num_clients)
    per_round_total = per_round_upload + per_round_download

    return {
        "shared_param_count": int(shared_param_count),
        "private_param_count": int(private_param_count),
        "total_tracked_param_count": int(total_param_count),
        "shared_param_ratio": float(shared_param_ratio),
        "shared_state_bytes": int(shared_bytes),
        "private_state_bytes": int(private_bytes),
        "total_tracked_state_bytes": int(total_bytes),
        "shared_state_ratio": float(shared_bytes / total_bytes) if total_bytes > 0 else 0.0,
        "num_clients": int(num_clients),
        "per_client_upload_bytes": int(shared_bytes),
        "per_client_download_bytes": int(shared_bytes),
        "per_round_upload_bytes": per_round_upload,
        "per_round_download_bytes": per_round_download,
        "per_round_total_bytes": per_round_total,
        "total_upload_bytes": int(per_round_upload * rounds),
        "total_download_bytes": int(per_round_download * rounds),
        "total_communication_bytes": int(per_round_total * rounds),
    }


def _detach_to_cpu(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    return value.cpu() if value.is_cuda else value.clone()


def snapshot_model_state(model) -> Dict[str, torch.Tensor]:
    return {
        key: _detach_to_cpu(value)
        for key, value in model.state_dict().items()
        if not is_redundant_backbone_alias_key(key)
    }


def snapshot_state_subset(model, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        key: _detach_to_cpu(state[key])
        for key in keys
        if key in state and not is_redundant_backbone_alias_key(key)
    }


def get_trainable_state_keys(model, keys: Iterable[str]) -> List[str]:
    """Restrict snapshots to keys with requires_grad=True.

    Frozen pretrained backbone weights never change and therefore do not need
    per-round snapshots or restores.
    """
    params = dict(model.named_parameters())
    return [key for key in keys if key in params and params[key].requires_grad]


def resolve_scope(args) -> Dict[str, object]:
    if getattr(args, "method", "") != "morphfl":
        raise ValueError(f"Unsupported method for MorphFL scope: {getattr(args, 'method', '')}")

    scope_mode = str(getattr(args, "shared_scope_mode", "default") or "default").strip().lower()
    if scope_mode not in {"default", "narrow", "wide"}:
        raise ValueError(f"Unsupported shared scope mode: {scope_mode}")

    shared_prefixes = [
        # VSP: rank-16 adapter, 64-d projector, context encoders, decision heads.
        "vsp.morph_adapter.",
        "vsp.concept_projector.",
        "vsp.context_encoder.",
        "vsp.local_summary_encoder.",
        "vsp.ns_gate.",
        "vsp.abn_cls.",
        "vsp.base_classifier.",
        # MEM: compiler, compiled classifier, scalar anchor, target predictors.
        "mem.compiler_head.",
        "mem.compiler_tail.",
        "mem.compiled_classifier.",
        "mem.anchor_classifier.",
        "mem.scalar_norm.",
        "mem.head_morph_predictor.",
        "mem.tail_morph_predictor.",
        "mem.head_evidence_encoder.",
        "mem.tail_evidence_encoder.",
        # URC: per-path reliability heads and the fusion-gate bias.
        "urc.measurement_head.",
        "urc.visual_head.",
        "urc.gate_bias",
    ]
    shared_parts = [
        "vsp",
        "mem_compiler",
        "mem_anchor",
        "decision_heads",
    ]
    if bool(getattr(args, "rho_enabled", False)):
        shared_parts.append("urc")
    private_parts = [
        "visual_frontend",
        "vsp_style_adapter",
        "vsp_style_calibrator",
    ]
    if bool(getattr(args, "share_global_context", True)):
        shared_parts = ["context_encoder", *shared_parts]

    if scope_mode == "narrow":
        shared_prefixes = [
            "mem.scalar_norm.",
            "mem.anchor_classifier.",
        ]
        shared_parts = ["scalar_norm", "anchor_classifier"]
        private_parts = [
            "vsp",
            "mem_compiler",
            "mem_anchor_remainder",
            "decision_heads",
            "urc",
            *private_parts,
        ]
    elif scope_mode == "wide":
        shared_prefixes = ["frontend.", *shared_prefixes]
        shared_parts = ["visual_frontend", *shared_parts]
        private_parts = [part for part in private_parts if part != "visual_frontend"]

    return {
        "shared_scope_mode": scope_mode,
        "shared_prefixes": tuple(shared_prefixes),
        "shared_scope": "+".join(shared_parts),
        "private_scope": "+".join(private_parts),
    }


def get_shared_private_keys(
    state_dict: Mapping[str, torch.Tensor],
    args=None,
) -> Tuple[List[str], List[str]]:
    keys = [key for key in state_dict.keys() if not is_redundant_backbone_alias_key(key)]
    shared_prefixes = resolve_scope(args)["shared_prefixes"]
    shared = [key for key in keys if key.startswith(shared_prefixes)]
    private = [key for key in keys if key not in shared]
    return shared, private


def load_state_subset(model, subset: Mapping[str, torch.Tensor]) -> None:
    """Load only the provided keys into the model; all other keys stay unchanged."""
    if not subset:
        return
    state = model.state_dict()
    payload: Dict[str, torch.Tensor] = {}
    for key, value in subset.items():
        if key not in state or is_redundant_backbone_alias_key(key):
            continue
        payload[key] = value
        alias_key = mirror_backbone_key(key)
        if alias_key is not None and alias_key in state:
            payload[alias_key] = value
    if payload:
        model.load_state_dict(payload, strict=False)


def average_states(
    client_states: List[Mapping[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    """Sample-count weighted average of client states."""
    total = float(sum(weights))
    averaged = {}
    for key in client_states[0].keys():
        ref = client_states[0][key]
        if not torch.is_floating_point(ref):
            averaged[key] = ref.clone()
            continue
        acc = None
        for state, weight in zip(client_states, weights):
            term = state[key].float() * (float(weight) / total)
            acc = term if acc is None else acc + term
        averaged[key] = acc.to(dtype=ref.dtype)
    return averaged
