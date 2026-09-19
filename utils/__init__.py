"""Utility subpackage: datasets, measurement pipeline, federation logic.

Heavy modules that import the model package (Federation, Evaluation) are
imported lazily to avoid an import cycle; import them from their submodule,
e.g. ``from utils.Federation import run_experiment``.
"""

from .Utils import build_arg_parser, set_seed, write_json, compute_metrics_multiclass
from .Datasets import (
    CLASS_TO_IDX,
    SIPAKMORPH_FEATURE_COLUMNS,
    SMIDSDataset,
    SIPaKMeDDataset,
    build_class_to_idx,
    collate_fn,
    parent_image_key,
)
from .DomainShift import apply_domain_shift_pil, canonicalize_domain_profile, has_domain_shift
from .Morphology import MorphologyExtractor
from .BoundaryScope import average_states, get_shared_private_keys, resolve_scope

__all__ = [
    "build_arg_parser", "set_seed", "write_json", "compute_metrics_multiclass",
    "CLASS_TO_IDX", "SIPAKMORPH_FEATURE_COLUMNS", "SMIDSDataset", "SIPaKMeDDataset",
    "build_class_to_idx", "collate_fn", "parent_image_key",
    "apply_domain_shift_pil", "canonicalize_domain_profile", "has_domain_shift",
    "MorphologyExtractor",
    "average_states", "get_shared_private_keys", "resolve_scope",
]
