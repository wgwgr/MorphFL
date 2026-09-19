"""SMIDS and SIPaKMeD federated datasets.

Each sample is returned as the six-tuple consumed by :func:`collate_fn`:

    (image, label, scalar_features, validity, relative_path, evidence_quality)

- ``image``: a PIL image resized to 224x224 (the domain profile and training
  augmentations are applied by the dataset/loader);
- ``scalar_features``: z-score normalized morphology scalars, with statistics
  estimated from the local training split only;
- ``validity``: per-scalar validity mask;
- ``evidence_quality``: external confidence of the measurement channel.

On SMIDS the two anchor scalars (head-ratio z-score, deviation from the WHO
band midpoint) are re-estimated from the presented image, so under a shift
profile the measurements degrade exactly as they would at deployment. On
SIPaKMeD the 34 scalars are read from the released official feature table
and are independent of the image.
"""


import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.DomainShift import apply_domain_shift_pil, canonicalize_domain_profile, has_domain_shift
from utils.Morphology import get_cache_paths, head_ratio_details


CLASS_TO_IDX = {"Normal_Sperm": 0, "Abnormal_Sperm": 1, "Non-Sperm": 2}

_MORPH_CACHE_BY_ROOT: Dict[str, Tuple[object, Dict[Tuple[str, str], int]]] = {}
_SIPAK_MORPH_CACHE_BY_ROOT: Dict[str, Tuple[object, Dict[str, int]]] = {}

SIPAK_NUCLEUS_SHAPE_COLUMNS = (
    "nucleus_area",
    "nucleus_major_axis_length",
    "nucleus_minor_axis_length",
    "nucleus_eccentricity",
    "nucleus_orientation_deg",
    "nucleus_equivalent_diameter",
    "nucleus_solidity",
    "nucleus_extent",
)
SIPAK_CYTOPLASM_SHAPE_COLUMNS = (
    "cytoplasm_area",
    "cytoplasm_major_axis_length",
    "cytoplasm_minor_axis_length",
    "cytoplasm_eccentricity",
    "cytoplasm_orientation_deg",
    "cytoplasm_equivalent_diameter",
    "cytoplasm_solidity",
    "cytoplasm_extent",
)
SIPAK_NUCLEUS_TEXTURE_COLUMNS = (
    "nucleus_r_avg_intensity",
    "nucleus_r_avg_contrast",
    "nucleus_r_entropy",
    "nucleus_g_avg_intensity",
    "nucleus_g_avg_contrast",
    "nucleus_g_entropy",
    "nucleus_b_avg_intensity",
    "nucleus_b_avg_contrast",
    "nucleus_b_entropy",
)
SIPAK_CYTOPLASM_TEXTURE_COLUMNS = (
    "cytoplasm_r_avg_intensity",
    "cytoplasm_r_avg_contrast",
    "cytoplasm_r_entropy",
    "cytoplasm_g_avg_intensity",
    "cytoplasm_g_avg_contrast",
    "cytoplasm_g_entropy",
    "cytoplasm_b_avg_intensity",
    "cytoplasm_b_avg_contrast",
    "cytoplasm_b_entropy",
)
SIPAKMORPH_FEATURE_COLUMNS = (
    *SIPAK_NUCLEUS_SHAPE_COLUMNS,
    *SIPAK_NUCLEUS_TEXTURE_COLUMNS,
    *SIPAK_CYTOPLASM_SHAPE_COLUMNS,
    *SIPAK_CYTOPLASM_TEXTURE_COLUMNS,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def reinhard_normalize(img: Image.Image, ref_mean, ref_std) -> Image.Image:
    """Align per-channel RGB mean/std to a training reference distribution.

    Applied to shifted images at evaluation time only. It is a per-channel
    linear remap and does not alter geometric structure; morphology scalars
    are unaffected because they come from precomputed measurements.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    src_mean = arr.reshape(-1, 3).mean(axis=0)
    src_std = arr.reshape(-1, 3).std(axis=0) + 1e-6
    normalized = (arr - src_mean) / src_std * np.asarray(ref_std, dtype=np.float32) + np.asarray(
        ref_mean, dtype=np.float32
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    return Image.fromarray((normalized * 255.0).astype(np.uint8))


def _load_rgb_image_with_retry(path: str, retries: int = 3, retry_delay: float = 0.1) -> Image.Image:
    """Robust image reader for transient I/O errors inside DataLoader workers."""
    last_error = None
    for attempt in range(retries):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except OSError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(retry_delay)

    try:
        with open(path, "rb") as f:
            buffer = f.read()
        with Image.open(io.BytesIO(buffer)) as img:
            return img.convert("RGB")
    except OSError:
        raise last_error


def _load_smids_morph_cache(data_dir: str) -> Tuple[object, Dict[Tuple[str, str], int]]:
    data_root = os.path.abspath(data_dir)
    cached = _MORPH_CACHE_BY_ROOT.get(data_root)
    if cached is not None:
        return cached

    cache_paths = get_cache_paths(data_root)
    cache_npz = cache_paths["morph_npz"]
    index_json = cache_paths["index_json"]

    if not os.path.exists(cache_npz) or not os.path.exists(index_json):
        cached = (False, {})
        _MORPH_CACHE_BY_ROOT[data_root] = cached
        return cached

    morph_features = dict(np.load(cache_npz, allow_pickle=True))
    with open(index_json) as f:
        index = json.load(f)
    morph_lookup = {}
    for i, rel_path in enumerate(index["paths"]):
        cls_name, fname = rel_path.split("/", 1)
        morph_lookup[(cls_name, fname)] = i
    cached = (morph_features, morph_lookup)
    _MORPH_CACHE_BY_ROOT[data_root] = cached
    return cached


def build_class_to_idx(class_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): idx for idx, name in enumerate(class_names)}


def parent_image_key(relpath: str) -> str:
    """Group key identifying the source acquisition image of a crop.

    SIPaKMeD single-cell crops are named ``<image_id>_<cell_id>.bmp`` inside a
    ``CROPPED`` directory; crops with the same image_id were cut from one
    cluster-cell image and must stay in the same partition. The key includes
    the class directory because image ids restart at 001 in every class.
    Datasets without cluster crops (SMIDS) use the full relpath, so each
    independent image forms its own group.
    """
    normalized = str(relpath).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[-2] == "CROPPED" and "_" in parts[-1]:
        stem = parts[-1].rsplit(".", 1)[0]
        image_id = stem.split("_", 1)[0]
        class_name = parts[-4] if len(parts) >= 4 else ""
        return f"{class_name}/{image_id}"
    return normalized


def infer_label_from_relpath(relpath: str, class_to_idx: Mapping[str, int]) -> str:
    for part in Path(relpath).parts:
        if part in class_to_idx:
            return part
    raise KeyError(f"Failed to infer class name from relpath: {relpath}")


def _build_sipak_relpath(class_name: str, image_id: str, cell_id: str) -> str:
    image_int = int(float(image_id))
    cell_int = int(float(cell_id))
    return f"images/{class_name}/im_{class_name}/CROPPED/{image_int:03d}_{cell_int:02d}.bmp"


def _load_sipak_morph_cache(data_dir: str) -> Tuple[object, Dict[str, int]]:
    data_root = os.path.abspath(data_dir)
    cached = _SIPAK_MORPH_CACHE_BY_ROOT.get(data_root)
    if cached is not None:
        return cached

    csv_path = os.path.join(data_root, "sipakmed_morphology_features.csv")
    if not os.path.exists(csv_path):
        cached = (False, {})
        _SIPAK_MORPH_CACHE_BY_ROOT[data_root] = cached
        return cached

    rows = []
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(row)
            relpath = _build_sipak_relpath(
                class_name=str(row["class_name"]),
                image_id=str(row["image_id"]),
                cell_id=str(row["cell_id"]),
            )
            lookup[relpath] = idx
    cached = (rows, lookup)
    _SIPAK_MORPH_CACHE_BY_ROOT[data_root] = cached
    return cached


class SMIDSDataset(Dataset):
    """SMIDS sperm dataset with the two head-ratio anchor scalars."""

    # SMIDS measurements are re-estimated per image and thus less reliable
    # than the released SIPaKMeD table; this scales their evidence strength.
    EVIDENCE_QUALITY = np.array([0.35], dtype=np.float32)

    # WHO (Kruger, 6th ed.) normal head ratio band [1.5, 2.0].
    WHO_RATIO_LO = 1.5
    WHO_RATIO_HI = 2.0
    WHO_RATIO_MID = 0.5 * (WHO_RATIO_LO + WHO_RATIO_HI)

    def __init__(
        self,
        data_dir: str,
        split_list: List[str],
        transform=None,
        morph_stats: tuple = None,
        domain_shift_profile: Optional[Dict[str, object]] = None,
        stain_ref_mean=None,
        stain_ref_std=None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.split_list = list(split_list)
        self.transform = transform
        self.image_size = 224
        self.domain_shift_profile = canonicalize_domain_profile(domain_shift_profile)
        self.domain_shift_active = has_domain_shift(self.domain_shift_profile)
        self.stain_ref_mean = None if stain_ref_mean is None else tuple(float(v) for v in stain_ref_mean)
        self.stain_ref_std = None if stain_ref_std is None else tuple(float(v) for v in stain_ref_std)

        self._morph_features, self._morph_lookup = _load_smids_morph_cache(self.data_dir)
        self._has_morph = self._morph_features is not False
        self._shifted_ratio_cache: Dict[str, Tuple[float, bool]] = {}

        # z-score statistics use only the current training split; test
        # datasets receive the training statistics through ``morph_stats``.
        if morph_stats is not None:
            self.morph_mean, self.morph_std = morph_stats
        elif self._has_morph:
            self.morph_mean, self.morph_std = self._compute_morph_stats()
        else:
            self.morph_mean, self.morph_std = 0.0, 1.0
        self.morph_stats = (self.morph_mean, self.morph_std)

    def _compute_morph_stats(self):
        if self.domain_shift_active:
            vals = []
            for relpath in self.split_list:
                ratio, valid = self._get_shifted_head_ratio(relpath)
                if valid:
                    vals.append(ratio)
            if len(vals) < 2:
                return 0.0, 1.0
            arr = np.asarray(vals, dtype=np.float32)
            return float(arr.mean()), float(arr.std() + 1e-6)

        vals = []
        for relpath in self.split_list:
            parts = relpath.split(os.sep, 1)
            if len(parts) != 2:
                continue
            gidx = self._morph_lookup.get((parts[0], parts[1]))
            if gidx is not None and bool(self._morph_features["head_ratio_valid"][gidx]):
                vals.append(float(self._morph_features["head_ratio"][gidx]))
        if len(vals) < 2:
            return 0.0, 1.0
        arr = np.asarray(vals, dtype=np.float32)
        return float(arr.mean()), float(arr.std() + 1e-6)

    def _get_shifted_head_ratio(self, relpath: str) -> Tuple[float, bool]:
        cached = self._shifted_ratio_cache.get(relpath)
        if cached is not None:
            return cached
        path = os.path.join(self.data_dir, relpath)
        if not os.path.isfile(path):
            cached = (0.0, False)
        else:
            img = _load_rgb_image_with_retry(path)
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            img = apply_domain_shift_pil(img, self.domain_shift_profile)
            bgr = np.asarray(img, dtype=np.uint8)[:, :, ::-1]
            details = head_ratio_details(bgr)
            cached = (float(details.get("ratio", 0.0)), bool(details.get("valid", 0.0)))
        self._shifted_ratio_cache[relpath] = cached
        return cached

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx: int):
        relpath = self.split_list[idx]
        cls_name, fname = relpath.split(os.sep, 1)
        label = CLASS_TO_IDX[cls_name]
        img = _load_rgb_image_with_retry(os.path.join(self.data_dir, relpath))
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if self.domain_shift_active:
            img = apply_domain_shift_pil(img, self.domain_shift_profile)
        if self.domain_shift_active and self.stain_ref_mean is not None and self.stain_ref_std is not None:
            img = reinhard_normalize(img, self.stain_ref_mean, self.stain_ref_std)
        if self.transform is not None:
            img = self.transform(img)

        scalar_feats, validity = self._get_morph_features(cls_name, fname, relpath=relpath)
        return (
            img,
            int(label),
            scalar_feats,
            validity,
            relpath,
            self.EVIDENCE_QUALITY.copy(),
        )

    def _get_morph_features(self, cls_name: str, fname: str, relpath: Optional[str] = None):
        if self.domain_shift_active and relpath is not None:
            raw, valid = self._get_shifted_head_ratio(relpath)
            if valid:
                z_score = (raw - self.morph_mean) / self.morph_std
                band_deviation = abs(raw - self.WHO_RATIO_MID)
                return (
                    np.array([z_score, band_deviation], dtype=np.float32),
                    np.array([True], dtype=bool),
                )
        if self._has_morph:
            gidx = self._morph_lookup.get((cls_name, fname))
            if gidx is not None and bool(self._morph_features["head_ratio_valid"][gidx]):
                raw = float(self._morph_features["head_ratio"][gidx])
                z_score = (raw - self.morph_mean) / self.morph_std
                band_deviation = abs(raw - self.WHO_RATIO_MID)
                return (
                    np.array([z_score, band_deviation], dtype=np.float32),
                    np.array([True], dtype=bool),
                )
        return (
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([False], dtype=bool),
        )


class SIPaKMeDDataset(Dataset):
    """SIPaKMeD cervical-cell dataset with the official table scalars."""

    EVIDENCE_QUALITY = np.array([1.0], dtype=np.float32)

    def __init__(
        self,
        data_dir: str,
        split_list: List[str],
        class_to_idx: Mapping[str, int],
        transform=None,
        morph_feature_columns: Optional[Sequence[str]] = None,
        morph_stats=None,
        domain_shift_profile: Optional[Dict[str, object]] = None,
        stain_ref_mean=None,
        stain_ref_std=None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.split_list = list(split_list)
        self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        self.transform = transform
        self.image_size = 224
        self.domain_shift_profile = canonicalize_domain_profile(domain_shift_profile)
        self.domain_shift_active = has_domain_shift(self.domain_shift_profile)
        self.stain_ref_mean = None if stain_ref_mean is None else tuple(float(v) for v in stain_ref_mean)
        self.stain_ref_std = None if stain_ref_std is None else tuple(float(v) for v in stain_ref_std)
        self.morph_feature_columns = tuple(str(col) for col in (morph_feature_columns or ()))
        self.scalar_feat_dim = int(len(self.morph_feature_columns))
        if self.scalar_feat_dim <= 0:
            raise ValueError("SIPaKMeDDataset requires non-empty morph_feature_columns.")
        self._morph_rows, self._morph_lookup = _load_sipak_morph_cache(self.data_dir)
        self._has_morph = self._morph_rows is not False
        if morph_stats is not None:
            mean_arr, std_arr = morph_stats
            self.morph_mean = np.asarray(mean_arr, dtype=np.float32)
            self.morph_std = np.asarray(std_arr, dtype=np.float32)
        elif self._has_morph:
            self.morph_mean, self.morph_std = self._compute_morph_stats()
        else:
            self.morph_mean = np.zeros((self.scalar_feat_dim,), dtype=np.float32)
            self.morph_std = np.ones((self.scalar_feat_dim,), dtype=np.float32)
        self.morph_stats = (self.morph_mean, self.morph_std)

    def _compute_morph_stats(self):
        values = []
        for relpath in self.split_list:
            row_idx = self._morph_lookup.get(relpath)
            if row_idx is None:
                continue
            row = self._morph_rows[row_idx]
            try:
                vec = np.asarray(
                    [float(row[col]) for col in self.morph_feature_columns],
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(vec)):
                values.append(vec)
        if not values:
            return (
                np.zeros((self.scalar_feat_dim,), dtype=np.float32),
                np.ones((self.scalar_feat_dim,), dtype=np.float32),
            )
        stacked = np.stack(values, axis=0)
        return stacked.mean(axis=0).astype(np.float32), (stacked.std(axis=0) + 1e-6).astype(np.float32)

    def _get_morph_features(self, relpath: str):
        if not self._has_morph:
            return (
                np.zeros((self.scalar_feat_dim,), dtype=np.float32),
                np.zeros((1,), dtype=bool),
            )
        row_idx = self._morph_lookup.get(relpath)
        if row_idx is None:
            return (
                np.zeros((self.scalar_feat_dim,), dtype=np.float32),
                np.zeros((1,), dtype=bool),
            )
        row = self._morph_rows[row_idx]
        try:
            raw = np.asarray(
                [float(row[col]) for col in self.morph_feature_columns],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            return (
                np.zeros((self.scalar_feat_dim,), dtype=np.float32),
                np.zeros((1,), dtype=bool),
            )
        if not np.all(np.isfinite(raw)):
            return (
                np.zeros((self.scalar_feat_dim,), dtype=np.float32),
                np.zeros((1,), dtype=bool),
            )
        feats = (raw - self.morph_mean) / self.morph_std
        return feats.astype(np.float32), np.array([True], dtype=bool)

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx: int):
        relpath = self.split_list[idx]
        class_name = infer_label_from_relpath(relpath, self.class_to_idx)
        label = self.class_to_idx[class_name]
        img = _load_rgb_image_with_retry(os.path.join(self.data_dir, relpath))
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if self.domain_shift_active:
            img = apply_domain_shift_pil(img, self.domain_shift_profile)
        if self.domain_shift_active and self.stain_ref_mean is not None and self.stain_ref_std is not None:
            img = reinhard_normalize(img, self.stain_ref_mean, self.stain_ref_std)
        if self.transform is not None:
            img = self.transform(img)
        scalar_feats, validity = self._get_morph_features(relpath)
        return (
            img,
            int(label),
            scalar_feats,
            validity,
            relpath,
            self.EVIDENCE_QUALITY.copy(),
        )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image, copy=True)
    tensor = torch.from_numpy(arr).to(torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1)
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def collate_fn(batch):
    """Collate the six-tuple dataset protocol into batched tensors."""
    images, labels = [], []
    scalar_list, valid_list, relpaths, evidence_qualities = [], [], [], []
    for img, label, scalar_feats, validity, relpath, evidence_quality in batch:
        images.append(pil_to_tensor(img))
        labels.append(label)
        scalar_list.append(torch.from_numpy(np.ascontiguousarray(scalar_feats)))
        valid_list.append(torch.from_numpy(np.ascontiguousarray(validity)))
        relpaths.append(relpath)
        evidence_qualities.append(torch.from_numpy(np.ascontiguousarray(evidence_quality)).float())
    return {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long),
        "scalar_feats": torch.stack(scalar_list),
        "validity_mask": torch.stack(valid_list),
        "relpath": list(relpaths),
        "evidence_quality": torch.stack(evidence_qualities),
    }
