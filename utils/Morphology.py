"""Client-local morphology measurement pipeline phi.

Extracts photometric-invariant geometric scalars together with
validity/quality metadata from each cell image:

- mask-based head/nucleus segmentation and WHO-band head-ratio statistics
  for SMIDS;
- bidirectional tail candidate search and a weak tail descriptor;
- per-dataset on-disk caches.

Under a domain-shift profile the measurements are re-estimated from the
degraded image (SMIDS); the SIPaKMeD scalars are read independently from the
released official feature table by the dataset class. The extractor runs
entirely on the client and is never aggregated.
"""


import hashlib
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.filters import meijering

from utils.DomainShift import (
    apply_domain_shift_bgr,
    canonicalize_domain_profile,
    domain_profile_cache_key,
    has_domain_shift,
)


EL3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# On-disk cache filenames released with the SMIDS preprocessing artifacts.
HEAD_MASK_CACHE_FILE = "fgmask_mecp_224.npy"
INDEX_CACHE_FILE = "index_mecp_224.json"
MORPH_FEATURES_CACHE_FILE = "morph_features.npz"
SMIDS_ANATOMY_CACHE_FILE = "smids_anatomy_cache_v2.npz"


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _normalize_relpath(relpath: str) -> str:
    relpath = str(relpath).replace("\\", "/").strip()
    while relpath.startswith("./"):
        relpath = relpath[2:]
    return relpath


def _safe_mask_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return 0.0
    return float(np.asarray(arr, dtype=np.float32)[mask].mean())


def _foreground_ratio_quality(fg_ratio: float) -> float:
    fg_ratio = float(fg_ratio)
    if fg_ratio <= 0.02 or fg_ratio >= 0.50:
        return 0.0
    if fg_ratio <= 0.12:
        return _clip01((fg_ratio - 0.02) / 0.10)
    return _clip01(1.0 - (fg_ratio - 0.12) / 0.38)


def estimate_image_quality(bgr: np.ndarray, fg_mask: np.ndarray = None) -> float:
    if bgr is None:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = _clip01(np.log1p(blur_var) / 6.0)
    contrast = _clip01(float(gray.std()) / 64.0)
    if fg_mask is not None and np.any(fg_mask > 0):
        fg_ratio = float((np.asarray(fg_mask) > 0).mean())
    else:
        otsu_thr, _ = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_ratio = float((gray <= float(otsu_thr)).mean())
    fg_quality = _foreground_ratio_quality(fg_ratio)
    return float(0.45 * sharpness + 0.35 * contrast + 0.20 * fg_quality)


def estimate_head_mask_quality(mask: np.ndarray) -> float:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(mask_u8.sum())
    if area < 16:
        return 0.0
    ellipse = fit_mask_ellipse(mask_u8)
    if ellipse is None:
        return 0.0
    (_, _), (v1, v2), _ = ellipse
    ellipse_area = np.pi * (v1 / 2.0) * (v2 / 2.0)
    coverage_quality = _clip01(1.0 - abs(area / max(ellipse_area, 1e-5) - 1.0))
    height, width = mask_u8.shape
    border_margin = max(4, int(round(0.03 * min(height, width))))
    border_touch_px = float(
        mask_u8[:border_margin, :].sum()
        + mask_u8[-border_margin:, :].sum()
        + mask_u8[:, :border_margin].sum()
        + mask_u8[:, -border_margin:].sum()
    )
    border_quality = _clip01(1.0 - 4.0 * border_touch_px / max(float(area), 1.0))
    elongation_quality = _clip01(3.2 / max(max(v1, v2) / max(min(v1, v2), 1e-5), 1e-5))
    return float(0.45 * coverage_quality + 0.30 * border_quality + 0.25 * elongation_quality)


def estimate_tail_quality(
    candidate_masks: np.ndarray,
    candidate_priors: np.ndarray,
    candidate_scores: np.ndarray,
    response: np.ndarray,
) -> float:
    masks = np.asarray(candidate_masks, dtype=np.float32)
    priors = np.asarray(candidate_priors, dtype=np.float32).reshape(-1)
    scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    if masks.size == 0 or priors.size == 0 or scores.size == 0:
        return 0.0
    best_idx = int(np.argmax(priors))
    best_mask = masks[best_idx] > 0
    if not np.any(best_mask):
        return 0.0
    response_mean = _clip01(_safe_mask_mean(response, best_mask))
    prior_quality = _clip01(float(priors[best_idx]))
    margin_quality = _clip01(0.5 + 0.5 * abs(float(priors[0] - priors[-1])))
    score_quality = _clip01(0.5 + 0.5 * np.tanh(float(scores[best_idx])))
    return float(0.40 * response_mean + 0.30 * prior_quality + 0.15 * margin_quality + 0.15 * score_quality)


def _cache_dir(data_dir: str) -> str:
    return os.path.join(os.path.abspath(data_dir), "cache")


def get_cache_paths(data_dir: str) -> Dict[str, str]:
    cache_dir = _cache_dir(data_dir)
    return {
        "cache_dir": cache_dir,
        "head_mask_npy": os.path.join(cache_dir, HEAD_MASK_CACHE_FILE),
        "index_json": os.path.join(cache_dir, INDEX_CACHE_FILE),
        "morph_npz": os.path.join(cache_dir, MORPH_FEATURES_CACHE_FILE),
        "smids_anatomy_npz": os.path.join(cache_dir, SMIDS_ANATOMY_CACHE_FILE),
    }


def _domain_profile_slug(domain_shift_profile=None) -> str:
    profile = canonicalize_domain_profile(domain_shift_profile)
    name = str(profile.get("name", "custom")).strip().lower().replace("-", "_")
    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name).strip("_")
    if safe_name and safe_name != "custom":
        return safe_name
    digest = hashlib.sha1(domain_profile_cache_key(profile).encode("utf-8")).hexdigest()[:12]
    return f"custom_{digest}"


def get_smids_anatomy_cache_path(data_dir: str, domain_shift_profile=None) -> str:
    cache_paths = get_cache_paths(data_dir)
    if not has_domain_shift(domain_shift_profile):
        return cache_paths["smids_anatomy_npz"]
    slug = _domain_profile_slug(domain_shift_profile)
    return os.path.join(cache_paths["cache_dir"], f"smids_anatomy_cache_v2_{slug}.npz")


def load_index_paths(data_dir: str) -> List[str]:
    cache_paths = get_cache_paths(data_dir)
    candidate = cache_paths["index_json"]
    if os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            idx_obj = json.load(f)
        return idx_obj.get("paths", idx_obj)

    data_dir = os.path.abspath(data_dir)
    paths: List[str] = []
    for cls_name in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(".bmp"):
                paths.append(f"{cls_name}/{fname}")
    return paths


def fit_ellipse_axis(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate the major axis of a binary mask via PCA."""
    coords = np.where(mask > 0)
    if len(coords[0]) < 5:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.0
    pts = np.stack(coords[::-1], axis=1).astype(np.float32)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, -1]
    major_axis = major_axis / (np.linalg.norm(major_axis) + 1e-8)
    angle_deg = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    return major_axis, center, angle_deg


def fit_mask_ellipse(mask: np.ndarray):
    """Fit an OpenCV ellipse directly from a binary mask."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < 5:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 5:
        return None
    return cv2.fitEllipse(contour)


class TailProposal:
    """Bidirectional tail candidate regions extending from a head-region mask."""

    def __init__(
        self,
        image_size: int = 224,
        grid_size: int = 14,
        root_overlap_px: float = 8.0,
        max_parallel: float = 128.0,
        base_half_width: float = 8.0,
        fan_tan: float = 1.0,
        score_temperature: float = 4.0,
        sigmas=(0.5, 1.0, 1.5, 2.0),
    ):
        self.image_size = int(image_size)
        self.grid_size = int(grid_size)
        self.root_overlap_px = float(root_overlap_px)
        self.max_parallel = float(max_parallel)
        self.base_half_width = float(base_half_width)
        self.fan_tan = float(fan_tan)
        self.score_temperature = float(score_temperature)
        self.sigmas = tuple(float(s) for s in sigmas)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def propose(self, bgr: np.ndarray, head_region_mask: np.ndarray) -> dict:
        head_mask = (head_region_mask > 0).astype(np.uint8)
        if head_mask.sum() < 5:
            zero_img = np.zeros_like(head_mask, dtype=np.float32)
            zero_patch = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            return {
                "valid": 0.0,
                "axis": np.array([1.0, 0.0], dtype=np.float32),
                "center_xy": np.array([0.0, 0.0], dtype=np.float32),
                "tip_neg_xy": np.array([0.0, 0.0], dtype=np.float32),
                "tip_pos_xy": np.array([0.0, 0.0], dtype=np.float32),
                "candidate_neg": zero_img,
                "candidate_pos": zero_img,
                "candidate_neg_patch": zero_patch,
                "candidate_pos_patch": zero_patch,
                "response": zero_img,
                "score_neg": 0.0,
                "score_pos": 0.0,
                "attn_neg": 0.5,
                "attn_pos": 0.5,
            }

        axis, center_xy, _ = fit_ellipse_axis(head_mask)
        axis = axis.astype(np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        pts = np.stack(np.where(head_mask > 0)[::-1], axis=1).astype(np.float32)
        proj = (pts - center_xy) @ axis
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)
        off = (pts - center_xy) @ perp
        proj_range = float(proj.max() - proj.min())
        body_half_width = float(np.percentile(np.abs(off), 90))
        dynamic_base_half_width = max(6.0, min(self.base_half_width, 0.9 * body_half_width + 2.0))
        dynamic_max_parallel = min(self.max_parallel, max(48.0, 1.35 * proj_range))
        dynamic_root_overlap = min(self.root_overlap_px, max(4.0, 0.20 * proj_range))
        tip_neg_xy = pts[int(np.argmin(proj))]
        tip_pos_xy = pts[int(np.argmax(proj))]

        response = self._ridge_response(bgr)
        cand_neg = self._build_candidate_mask(
            head_mask, tip_neg_xy, -axis, dynamic_root_overlap, dynamic_max_parallel, dynamic_base_half_width
        )
        cand_pos = self._build_candidate_mask(
            head_mask, tip_pos_xy, axis, dynamic_root_overlap, dynamic_max_parallel, dynamic_base_half_width
        )
        score_neg = self._score_candidate(cand_neg, response, tip_neg_xy, -axis, dynamic_max_parallel)
        score_pos = self._score_candidate(cand_pos, response, tip_pos_xy, axis, dynamic_max_parallel)
        attn_neg, attn_pos = self._softmax_pair(score_neg, score_pos)

        return {
            "valid": 1.0,
            "axis": axis,
            "center_xy": center_xy.astype(np.float32),
            "tip_neg_xy": tip_neg_xy.astype(np.float32),
            "tip_pos_xy": tip_pos_xy.astype(np.float32),
            "candidate_neg": cand_neg.astype(np.float32),
            "candidate_pos": cand_pos.astype(np.float32),
            "candidate_neg_patch": self._downsample_binary(cand_neg),
            "candidate_pos_patch": self._downsample_binary(cand_pos),
            "response": response.astype(np.float32),
            "score_neg": float(score_neg),
            "score_pos": float(score_pos),
            "attn_neg": float(attn_neg),
            "attn_pos": float(attn_pos),
        }

    def _build_candidate_mask(
        self,
        head_mask: np.ndarray,
        tip_xy: np.ndarray,
        axis_xy: np.ndarray,
        root_overlap_px: float,
        max_parallel: float,
        base_half_width: float,
    ) -> np.ndarray:
        h, w = head_mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - float(tip_xy[0])
        dy = yy - float(tip_xy[1])

        axis_xy = axis_xy.astype(np.float32)
        axis_xy = axis_xy / (np.linalg.norm(axis_xy) + 1e-8)
        perp_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float32)

        para = dx * axis_xy[0] + dy * axis_xy[1]
        off = dx * perp_xy[0] + dy * perp_xy[1]
        width = base_half_width + self.fan_tan * np.maximum(para, 0.0)
        candidate = (
            (para >= -root_overlap_px)
            & (para <= max_parallel)
            & (np.abs(off) <= width)
            & ((head_mask == 0) | (para <= 0.0))
        )
        return candidate.astype(np.uint8)

    def _ridge_response(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        value = self._clahe.apply(hsv[:, :, 2])
        value_f = value.astype(np.float32) / 255.0
        resp = meijering(value_f, sigmas=self.sigmas, black_ridges=True)
        resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        resp -= float(resp.min())
        peak = float(resp.max())
        if peak > 1e-8:
            resp /= peak
        return resp

    def _score_candidate(
        self,
        candidate_mask: np.ndarray,
        response: np.ndarray,
        tip_xy: np.ndarray,
        axis_xy: np.ndarray,
        max_parallel: float,
    ) -> float:
        if candidate_mask.sum() == 0:
            return 0.0

        h, w = candidate_mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - float(tip_xy[0])
        dy = yy - float(tip_xy[1])
        para = dx * axis_xy[0] + dy * axis_xy[1]

        weighted = response * candidate_mask.astype(np.float32)
        total_w = float(weighted.sum())
        if total_w <= 1e-8:
            return 0.0

        mean_resp = total_w / float(candidate_mask.sum())
        max_resp = float(weighted.max())
        thr = max(0.15, 0.35 * max_resp)
        extent = 0.0
        hit = (weighted >= thr) & (candidate_mask > 0)
        if np.any(hit):
            extent = float(np.clip(para[hit].max() / max_parallel, 0.0, 1.0))

        return 0.40 * mean_resp + 0.40 * max_resp + 0.20 * extent

    def _softmax_pair(self, score_a: float, score_b: float):
        scores = np.asarray([score_a, score_b], dtype=np.float32) * self.score_temperature
        scores -= float(scores.max())
        weights = np.exp(scores)
        weights /= float(weights.sum() + 1e-8)
        return float(weights[0]), float(weights[1])

    def _downsample_binary(self, mask: np.ndarray) -> np.ndarray:
        block = self.image_size // self.grid_size
        reshaped = mask.reshape(self.grid_size, block, self.grid_size, block)
        pooled = reshaped.max(axis=(1, 3)).astype(np.float32)
        return pooled


class SmidsAnatomyCache:
    """Memory-mapped view of the precomputed SMIDS anatomy cache."""

    def __init__(self, data_dir: str, domain_shift_profile=None):
        self.data_dir = os.path.abspath(data_dir)
        self.domain_shift_profile = canonicalize_domain_profile(domain_shift_profile)
        self.available = False
        self.lookup: Dict[str, int] = {}
        self.cache = None
        cache_path = get_smids_anatomy_cache_path(self.data_dir, domain_shift_profile=self.domain_shift_profile)
        if not os.path.exists(cache_path):
            return
        self.cache = np.load(cache_path, allow_pickle=True)
        paths = [_normalize_relpath(rel) for rel in self.cache["paths"].tolist()]
        self.lookup = {rel: i for i, rel in enumerate(paths)}
        self.available = True

    def lookup_index(self, relpath: str) -> int:
        if not self.available or self.cache is None:
            raise FileNotFoundError("SMIDS anatomy cache is unavailable for the current data/domain profile.")
        relpath = _normalize_relpath(relpath)
        idx = self.lookup.get(relpath)
        if idx is None:
            raise KeyError(f"SMIDS anatomy cache lookup miss for relpath {relpath!r}.")
        return int(idx)

    def batch_lookup_indices(self, relpaths: List[str]) -> np.ndarray:
        return np.asarray([self.lookup_index(rel) for rel in relpaths], dtype=np.int64)


def segment_head_region(bgr_img: np.ndarray) -> Dict[str, np.ndarray]:
    """Segment the sperm head region from a resized BGR image."""

    def _ellipse_fill(mask_shape, ellipse) -> np.ndarray:
        fill = np.zeros(mask_shape, dtype=np.uint8)
        if ellipse is None:
            return fill
        cv2.ellipse(fill, ellipse, 1, thickness=-1, lineType=cv2.LINE_AA)
        return fill

    def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
        binary_mask = (binary_mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            return binary_mask
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if n_comp <= 1:
            return binary_mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest).astype(np.uint8)

    def _smooth_mask(binary_mask: np.ndarray, close_radius: int, open_radius: int) -> np.ndarray:
        mask = _largest_component(binary_mask)
        if mask.sum() == 0:
            return mask
        if close_radius >= 3:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius, close_radius))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        if open_radius >= 3:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius, open_radius))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        return _largest_component(mask)

    def _ellipse_major_unit(axis_a: float, axis_b: float, angle_deg: float) -> np.ndarray:
        theta = np.deg2rad(angle_deg)
        major_unit = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        if axis_b > axis_a:
            major_unit = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        norm = float(np.linalg.norm(major_unit))
        if norm <= 1e-6:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return major_unit / norm

    def _build_front_cap_mask(
        mask_shape,
        center_xy,
        axes_xy,
        angle_deg: float,
        anchor_xy,
        min_proj_ratio: float = -0.20,
    ) -> np.ndarray:
        major_radius = max(float(max(axes_xy)), 1.0)
        axis_dir = _ellipse_major_unit(float(axes_xy[0]), float(axes_xy[1]), angle_deg)
        if anchor_xy is not None:
            front_vec = np.asarray(
                [float(center_xy[0]) - float(anchor_xy[0]), float(center_xy[1]) - float(anchor_xy[1])],
                dtype=np.float32,
            )
            if float(np.dot(front_vec, axis_dir)) < 0.0:
                axis_dir = -axis_dir
        yy, xx = np.indices(mask_shape, dtype=np.float32)
        proj = (xx - float(center_xy[0])) * axis_dir[0] + (yy - float(center_xy[1])) * axis_dir[1]
        return (proj >= (min_proj_ratio * major_radius)).astype(np.uint8)

    def _build_head_focus_mask(component_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        component_mask = (component_mask > 0).astype(np.uint8)
        if component_mask.sum() <= 4:
            return component_mask, 0.0
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        peak = float(dist.max())
        if peak <= 1e-6:
            return component_mask, 0.0

        seed_hi = 0.55
        seed_lo = 0.40
        kernel_scale = 2.2
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and len(cnts[0]) >= 5:
            try:
                comp_ell = cv2.fitEllipse(cnts[0])
                (_, _), (cv1, cv2_len), _ = comp_ell
                comp_ratio = max(cv1, cv2_len) / max(min(cv1, cv2_len), 1e-5)
                if comp_ratio > 3.0:
                    seed_hi = 0.64
                    seed_lo = 0.48
                    kernel_scale = 1.7
            except cv2.error:
                pass

        seed = (dist >= max(1.0, seed_hi * peak)).astype(np.uint8)
        if seed.sum() == 0:
            seed = (dist >= max(0.8, seed_lo * peak)).astype(np.uint8)
        if seed.sum() == 0:
            return component_mask, peak
        seed = _largest_component(seed)

        kernel_size = max(7, min(31, int(round(kernel_scale * peak)) * 2 + 1))
        support = cv2.dilate(
            seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )
        focus = np.logical_and(component_mask > 0, support > 0).astype(np.uint8)
        focus = _largest_component(focus)
        if focus.sum() < max(30, int(0.18 * component_mask.sum())):
            return component_mask, peak
        return focus, peak

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fgo = cv2.morphologyEx(fg, cv2.MORPH_OPEN, EL3, 1)
    n, lab, st, cent = cv2.connectedComponentsWithStats(fgo, 8)
    height, width = gray.shape
    cx0, cy0 = width / 2, height / 2
    diag = np.hypot(cx0, cy0)

    best = -1.0
    best_obj = None
    for idx in range(1, n):
        area = int(st[idx, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        comp = (lab == idx).astype(np.uint8)
        head_focus, peak = _build_head_focus_mask(comp)
        cnts, _ = cv2.findContours(head_focus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or len(cnts[0]) < 5:
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts or len(cnts[0]) < 5:
                continue
            head_focus = comp
        contour = cnts[0]
        try:
            ell = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        focus_area = max(int(head_focus.sum()), 1)
        (_, _), (v1, v2), _ = ell
        ellipse_ratio = max(v1, v2) / max(min(v1, v2), 1e-5)
        ellipse_area = np.pi * (v1 / 2.0) * (v2 / 2.0)
        coverage = focus_area / max(ellipse_area, 1e-5)
        ellipse_coverage_quality = _clip01(1.0 - abs(coverage - 1.0))
        focus_ys, focus_xs = np.where(head_focus > 0)
        if focus_ys.size > 0:
            focus_cx = float(focus_xs.mean())
            focus_cy = float(focus_ys.mean())
        else:
            focus_cx, focus_cy = float(cent[idx][0]), float(cent[idx][1])
        center_dist = np.hypot(focus_cx - cx0, focus_cy - cy0) / max(diag, 1e-6)
        center_quality = _clip01(1.0 - center_dist)
        elongation_quality = _clip01(3.2 / max(ellipse_ratio, 1e-5))
        border_margin = max(4, int(round(0.03 * min(height, width))))
        x, y, w_box, h_box = cv2.boundingRect(contour)
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + w_box) >= (width - border_margin)
            or (y + h_box) >= (height - border_margin)
        )
        border_touch_px = float(
            comp[:border_margin, :].sum()
            + comp[-border_margin:, :].sum()
            + comp[:, :border_margin].sum()
            + comp[:, -border_margin:].sum()
        )
        border_touch_ratio = border_touch_px / max(float(area), 1.0)
        border_quality = _clip01(1.0 - 4.0 * border_touch_ratio)
        if touches_border:
            border_quality *= 0.35
        border_quality = max(border_quality, 0.05)
        seed_quality = _clip01(peak / max(0.5 * min(height, width), 1e-5))

        score = (
            focus_area
            * ellipse_coverage_quality
            * max(center_quality, 0.15)
            * max(border_quality, 0.10)
            * max(elongation_quality, 0.20)
            * max(0.7 + 0.3 * seed_quality, 0.1)
        )
        if score > best:
            best = score
            best_obj = {
                "component_mask": comp,
                "head_focus_mask": head_focus,
                "ellipse": ell,
                "area": focus_area,
                "ellipse_coverage_quality": ellipse_coverage_quality,
                "center_quality": center_quality,
                "border_quality": border_quality,
                "elongation_quality": elongation_quality,
                "peak": peak,
            }

    if best_obj is None:
        return {
            "head_mask": np.zeros_like(gray, dtype=np.float32),
            "ratio": 0.0,
            "valid": 0.0,
        }

    (_, _), (v1, v2), _ = best_obj["ellipse"]
    ratio = max(v1, v2) / max(min(v1, v2), 1e-5)
    component_mask = best_obj["component_mask"].astype(np.uint8)
    head_focus_mask = best_obj["head_focus_mask"].astype(np.uint8)
    ellipse_fill = _ellipse_fill(gray.shape, best_obj["ellipse"])
    dist = cv2.distanceTransform(head_focus_mask, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak > 1e-6:
        seed = (dist >= max(1.0, 0.50 * peak)).astype(np.uint8)
        if seed.sum() == 0:
            seed = (dist >= max(0.8, 0.35 * peak)).astype(np.uint8)
        seed = _largest_component(seed)
    else:
        seed = np.zeros_like(head_focus_mask, dtype=np.uint8)

    if seed.sum() > 0:
        ys, xs = np.where(seed > 0)
        seed_center = (float(xs.mean()), float(ys.mean()))
    else:
        seed_center = tuple(float(v) for v in best_obj["ellipse"][0])

    major_len = max(v1, v2)
    minor_len = min(v1, v2)
    local_major = min(0.55 * major_len, max(1.55 * minor_len, 4.0 * peak, 16.0))
    local_minor = min(1.10 * minor_len, max(0.90 * minor_len, 3.0 * peak, 9.0))
    if v1 >= v2:
        local_axes = (local_major, local_minor)
    else:
        local_axes = (local_minor, local_major)
    seed_ellipse = _ellipse_fill(
        gray.shape,
        (seed_center, tuple(float(v) for v in local_axes), best_obj["ellipse"][2]),
    )
    dilate_radius = max(5, min(31, int(round(2.5 * max(peak, 1.0))) * 2 + 1))
    seed_support = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius, dilate_radius)),
        iterations=1,
    )
    support_mask = np.logical_or(seed_support > 0, seed_ellipse > 0)
    geometry_mask = np.logical_and(
        np.logical_or(head_focus_mask > 0, ellipse_fill > 0),
        support_mask,
    ).astype(np.uint8)
    if geometry_mask.sum() < max(30, int(0.25 * head_focus_mask.sum())):
        geometry_mask = np.logical_or(
            np.logical_and(head_focus_mask > 0, seed_support > 0),
            np.logical_and(ellipse_fill > 0, seed_ellipse > 0),
        ).astype(np.uint8)
    expanded_axes = (1.12 * float(local_axes[0]), 1.12 * float(local_axes[1]))
    expanded_ellipse = _ellipse_fill(
        gray.shape,
        (seed_center, expanded_axes, best_obj["ellipse"][2]),
    )
    component_ys, component_xs = np.where(component_mask > 0)
    if component_ys.size > 0:
        component_center = (float(component_xs.mean()), float(component_ys.mean()))
    else:
        component_center = tuple(float(v) for v in best_obj["ellipse"][0])
    front_cap_axes = (
        1.34 * float(max(local_axes)),
        1.14 * float(min(local_axes)),
    )
    front_cap_ellipse = _ellipse_fill(
        gray.shape,
        (seed_center, front_cap_axes, best_obj["ellipse"][2]),
    )
    front_cap_region = np.logical_and(
        front_cap_ellipse > 0,
        _build_front_cap_mask(
            gray.shape, seed_center, front_cap_axes, best_obj["ellipse"][2], component_center, -0.12
        )
        > 0,
    ).astype(np.uint8)
    front_anchor_radius = max(3, min(11, int(round(0.90 * max(peak, 1.0))) * 2 + 1))
    front_anchor = cv2.dilate(
        component_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (front_anchor_radius, front_anchor_radius)),
        iterations=1,
    )
    front_cap_component = np.logical_and(front_cap_region > 0, front_anchor > 0).astype(np.uint8)
    front_cap_enabled = (
        ratio <= 2.35
        and best_obj["ellipse_coverage_quality"] >= 0.76
        and best_obj["border_quality"] >= 0.38
        and int(component_mask.sum()) >= int(head_focus_mask.sum()) + max(120, int(0.08 * component_mask.sum()))
    )
    if (
        1.10 <= ratio <= 2.20
        and 1500 <= int(head_focus_mask.sum()) <= 3800
        and best_obj["ellipse_coverage_quality"] >= 0.82
        and best_obj["border_quality"] >= 0.40
        and best_obj["elongation_quality"] >= 0.95
    ):
        relax_radius = max(3, min(11, int(round(1.2 * max(peak, 1.0))) * 2 + 1))
        relaxed = cv2.dilate(
            geometry_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (relax_radius, relax_radius)),
            iterations=1,
        )
        relax_mask = np.logical_and(relaxed > 0, np.logical_or(component_mask > 0, expanded_ellipse > 0))
        geometry_mask = np.logical_or(geometry_mask > 0, relax_mask).astype(np.uint8)
    if front_cap_enabled:
        geometry_mask = np.logical_or(geometry_mask > 0, front_cap_component > 0).astype(np.uint8)
    component_area = int(component_mask.sum())
    geometry_area = int(geometry_mask.sum())
    if (
        1.20 <= ratio <= 1.85
        and component_area > geometry_area + 120
        and best_obj["ellipse_coverage_quality"] >= 0.80
        and best_obj["border_quality"] >= 0.40
    ):
        retained_component = np.logical_and(component_mask > 0, expanded_ellipse > 0)
        geometry_mask = np.logical_or(geometry_mask > 0, retained_component).astype(np.uint8)
    if (
        ratio <= 2.30
        and 1100 <= int(geometry_mask.sum()) <= 4600
        and best_obj["ellipse_coverage_quality"] >= 0.78
        and best_obj["border_quality"] >= 0.42
    ):
        refine_radius = max(5, min(15, int(round(1.6 * max(peak, 1.0))) * 2 + 1))
        refine_support = cv2.dilate(
            geometry_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_radius, refine_radius)),
            iterations=1,
        )
        if front_cap_enabled:
            front_support = cv2.dilate(
                front_cap_component,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, refine_radius - 2), max(5, refine_radius - 2))),
                iterations=1,
            )
        else:
            front_support = np.zeros_like(refine_support)
        refine_envelope = np.logical_or(expanded_ellipse > 0, front_cap_region > 0)
        refine_support = np.logical_and(np.logical_or(refine_support > 0, front_support > 0), refine_envelope).astype(
            np.uint8
        )
        refine_vals = gray[refine_support > 0]
        if refine_vals.size >= 64:
            local_thr, _ = cv2.threshold(
                refine_vals.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            gray_slack = 24.0 if ratio <= 1.85 else 34.0
            gray_candidate = gray <= min(255.0, float(local_thr) + gray_slack)
            core_sat_vals = sat[geometry_mask > 0]
            core_val_vals = val[geometry_mask > 0]
            if core_sat_vals.size > 0:
                sat_thr = max(12.0, float(np.percentile(core_sat_vals, 15)) - 18.0)
            else:
                sat_thr = 12.0
            sat_candidate = sat >= sat_thr
            # Recover pale acrosome caps that are slightly brighter and less
            # saturated than the core but darker than the local background ring.
            if core_val_vals.size > 0:
                pale_val_lo = max(0.0, float(np.percentile(core_val_vals, 35)) - 8.0)
                pale_val_hi = min(255.0, float(np.percentile(core_val_vals, 92)) + 12.0)
            else:
                pale_val_lo, pale_val_hi = 96.0, 232.0
            support_ring = np.logical_and(refine_support > 0, geometry_mask == 0)
            ring_gray_vals = gray[support_ring]
            if ring_gray_vals.size > 0:
                pale_gray_hi = min(248.0, float(np.percentile(ring_gray_vals, 72)) - 2.0)
            else:
                pale_gray_hi = min(248.0, float(local_thr) + gray_slack + 10.0)
            if core_sat_vals.size > 0:
                pale_sat_hi = max(18.0, float(np.percentile(core_sat_vals, 88)) + 10.0)
            else:
                pale_sat_hi = 72.0
            pale_candidate = np.logical_and.reduce(
                [
                    refine_support > 0,
                    gray <= pale_gray_hi,
                    val >= pale_val_lo,
                    val <= pale_val_hi,
                    sat <= pale_sat_hi,
                ]
            )
            refine_candidate = np.logical_and(
                refine_support > 0,
                np.logical_or(np.logical_or(gray_candidate, sat_candidate), pale_candidate),
            )
            geometry_mask = np.logical_or(geometry_mask > 0, refine_candidate).astype(np.uint8)

    # Prefer boundary rounding over aggressive shrinkage so recovered pale
    # edges are retained by the light opening step.
    smooth_close = max(5, min(13, int(round(1.0 * max(peak, 1.0))) * 2 + 1))
    smooth_open = max(3, min(5, int(round(0.30 * max(peak, 1.0))) * 2 + 1))
    presmooth_area = int(geometry_mask.sum())
    if (
        1.20 <= ratio <= 1.85
        and component_area >= 1400
        and presmooth_area < int(0.94 * component_area)
        and best_obj["ellipse_coverage_quality"] >= 0.80
        and best_obj["border_quality"] >= 0.40
    ):
        smooth_open = 1
    geometry_mask = _smooth_mask(geometry_mask, close_radius=smooth_close, open_radius=smooth_open)
    return {
        "head_mask": geometry_mask.astype(np.float32),
        "ratio": float(ratio),
        "valid": 1.0,
    }


def head_ratio_details(bgr_img: np.ndarray) -> Dict[str, float]:
    """Estimate the head ellipse ratio and its validity from a resized BGR image."""
    details = segment_head_region(bgr_img)
    return {
        "ratio": float(details["ratio"]),
        "valid": float(details["valid"]),
    }


def build_empty_measurement_item(image_size: int) -> Dict[str, np.ndarray]:
    return {
        "head_mask": np.zeros((image_size, image_size), dtype=np.float32),
        "head_valid": np.asarray([0.0], dtype=np.float32),
        "head_ratio": np.asarray([0.0], dtype=np.float32),
        "head_ratio_valid": np.asarray([0.0], dtype=np.float32),
        "image_quality": np.asarray([0.0], dtype=np.float32),
        "head_quality": np.asarray([0.0], dtype=np.float32),
        "candidate_masks": np.zeros((2, image_size, image_size), dtype=np.float32),
        "candidate_priors": np.asarray([0.5, 0.5], dtype=np.float32),
        "candidate_scores": np.zeros(2, dtype=np.float32),
        "tail_quality": np.asarray([0.0], dtype=np.float32),
        "tail_valid": np.asarray([0.0], dtype=np.float32),
        "tail_response": np.zeros((image_size, image_size), dtype=np.float32),
        "tail_axis": np.asarray([1.0, 0.0], dtype=np.float32),
        "tail_tip_neg_xy": np.zeros(2, dtype=np.float32),
        "tail_tip_pos_xy": np.zeros(2, dtype=np.float32),
    }


def build_measurement_item(
    head_mask: np.ndarray,
    ratio_details: Dict[str, np.ndarray],
    bgr: np.ndarray = None,
    proposer: TailProposal = None,
) -> Dict[str, np.ndarray]:
    image_size = int(head_mask.shape[0])
    item = build_empty_measurement_item(image_size=image_size)
    head_mask = (np.asarray(head_mask) > 0).astype(np.uint8)
    if head_mask.sum() <= 4:
        item["head_mask"] = head_mask.astype(np.float32)
        return item

    ratio = float(ratio_details.get("ratio", 0.0))
    ratio_valid = bool(ratio_details.get("valid", 0.0))
    head_valid = float(head_mask.sum() > 4)

    item["head_mask"] = head_mask.astype(np.float32)
    item["head_valid"] = np.asarray([head_valid], dtype=np.float32)
    item["head_ratio"] = np.asarray([ratio], dtype=np.float32)
    item["head_ratio_valid"] = np.asarray([float(ratio_valid)], dtype=np.float32)
    item["image_quality"] = np.asarray([estimate_image_quality(bgr, fg_mask=head_mask)], dtype=np.float32)
    item["head_quality"] = np.asarray([estimate_head_mask_quality(head_mask)], dtype=np.float32)

    if bgr is not None and proposer is not None:
        proposal = proposer.propose(bgr, head_mask)
        if float(proposal["valid"]) > 0.0:
            item["candidate_masks"] = np.stack(
                [proposal["candidate_neg"], proposal["candidate_pos"]],
                axis=0,
            ).astype(np.float32)
            item["candidate_priors"] = np.asarray(
                [proposal["attn_neg"], proposal["attn_pos"]],
                dtype=np.float32,
            )
            item["candidate_scores"] = np.asarray(
                [proposal["score_neg"], proposal["score_pos"]],
                dtype=np.float32,
            )
            item["tail_valid"] = np.asarray([1.0], dtype=np.float32)
            item["tail_response"] = proposal["response"].astype(np.float32)
            item["tail_axis"] = proposal["axis"].astype(np.float32)
            item["tail_tip_neg_xy"] = proposal["tip_neg_xy"].astype(np.float32)
            item["tail_tip_pos_xy"] = proposal["tip_pos_xy"].astype(np.float32)
            item["tail_quality"] = np.asarray(
                [
                    estimate_tail_quality(
                        item["candidate_masks"],
                        item["candidate_priors"],
                        item["candidate_scores"],
                        item["tail_response"],
                    )
                ],
                dtype=np.float32,
            )
    return item


class MorphologyExtractor:
    """Head mask, head ratio and tail proposal evidence for one dataset."""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        sigmas=(0.5, 1.0, 1.5, 2.0),
        fan_tan: float = 2.5,
        base_half_width: float = 10.0,
        max_parallel: float = 160.0,
        near_root_px: float = 24.0,
        domain_shift_profile=None,
        max_cache_items: int = 512,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.image_size = int(image_size)
        self.near_root_px = float(near_root_px)
        self.domain_shift_profile = canonicalize_domain_profile(domain_shift_profile)
        self.domain_shift_active = has_domain_shift(self.domain_shift_profile)

        cache_paths = get_cache_paths(self.data_dir)
        head_mask_npy = cache_paths["head_mask_npy"]
        morph_npz = cache_paths["morph_npz"]

        paths = [_normalize_relpath(rel) for rel in load_index_paths(self.data_dir)]
        self.lookup = {rel: i for i, rel in enumerate(paths)}
        self.class_lookup = {
            rel: rel.split("/", 1)[0] if "/" in rel else ""
            for rel in paths
        }
        self.head_masks = np.load(head_mask_npy, mmap_mode="r")

        self.morph_cache = np.load(morph_npz) if os.path.exists(morph_npz) else None
        self.proposer = TailProposal(
            image_size=self.image_size,
            sigmas=sigmas,
            fan_tan=fan_tan,
            base_half_width=base_half_width,
            max_parallel=max_parallel,
            root_overlap_px=max(4.0, 0.4 * near_root_px),
        )
        self.max_cache_items = max(int(max_cache_items), 0)
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def _store_cached_item(self, relpath: str, item: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cached_item = {k: v.astype(np.float32, copy=True) for k, v in item.items()}
        self._cache[relpath] = cached_item
        self._cache.move_to_end(relpath)
        if self.max_cache_items > 0 and len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)
        return {k: v.copy() for k, v in cached_item.items()}

    def extract(self, relpath: str) -> Dict[str, np.ndarray]:
        relpath = _normalize_relpath(relpath)
        cached = self._cache.get(relpath)
        if cached is not None:
            self._cache.move_to_end(relpath)
            return {k: v.copy() for k, v in cached.items()}

        class_name = self.class_lookup.get(relpath)
        if class_name == "Non-Sperm":
            item = self._zero_item()
            bgr = self._load_bgr(relpath)
            item["image_quality"] = np.asarray([estimate_image_quality(bgr)], dtype=np.float32)
            return self._store_cached_item(relpath, item)

        if self.domain_shift_active:
            bgr = self._load_bgr(relpath)
            if bgr is None:
                item = self._zero_item()
            else:
                head_details = segment_head_region(bgr)
                item = build_measurement_item(
                    head_mask=head_details["head_mask"],
                    ratio_details=head_details,
                    bgr=bgr,
                    proposer=self.proposer,
                )
            return self._store_cached_item(relpath, item)

        idx = self.lookup.get(relpath)
        if idx is None:
            raise KeyError(
                "Morphology cache lookup miss for relpath "
                f"{relpath!r}. Rebuild the cache/index or check split paths."
            )
        head_mask = (self.head_masks[idx] > 127).astype(np.uint8)
        if head_mask.sum() <= 4:
            item = self._zero_item()
            item["head_mask"] = head_mask.astype(np.float32)
            bgr = self._load_bgr(relpath)
            item["image_quality"] = np.asarray(
                [estimate_image_quality(bgr, fg_mask=head_mask)], dtype=np.float32
            )
        else:
            bgr = self._load_bgr(relpath)
            ratio_details = self._get_head_ratio_details(idx, bgr)
            item = build_measurement_item(
                head_mask=head_mask,
                ratio_details=ratio_details,
                bgr=bgr,
                proposer=self.proposer,
            )

        return self._store_cached_item(relpath, item)

    def _load_bgr(self, relpath: str):
        img_path = os.path.join(self.data_dir, relpath)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return apply_domain_shift_bgr(bgr, self.domain_shift_profile)

    def _get_head_ratio_details(self, idx: int, bgr: np.ndarray):
        if self.morph_cache is not None:
            ratios = self.morph_cache["head_ratio"]
            valids = self.morph_cache["head_ratio_valid"]
            if idx < len(ratios):
                return {
                    "ratio": float(ratios[idx]),
                    "valid": float(bool(valids[idx])),
                }
        if bgr is None:
            return {"ratio": 0.0, "valid": 0.0}
        return head_ratio_details(bgr)

    def _zero_item(self) -> Dict[str, np.ndarray]:
        return build_empty_measurement_item(image_size=self.image_size)


class HeadRegionExtractor:
    """Full head masks and head-level quality from the morphology extractor."""

    def __init__(self, data_dir: str, image_size: int = 224, extractor=None):
        self.extractor = extractor if extractor is not None else MorphologyExtractor(data_dir, image_size=image_size)

    def extract(self, relpath: str):
        evidence = self.extractor.extract(relpath)
        head_mask = evidence["head_mask"].astype(np.float32, copy=True)
        head_valid = np.asarray([float(head_mask.sum() > 4.0)], dtype=np.float32)
        return {
            "head_mask": head_mask,
            "valid": head_valid,
            "head_quality": evidence["head_quality"].astype(np.float32, copy=True),
            "image_quality": evidence["image_quality"].astype(np.float32, copy=True),
        }


class TailDescriptor:
    """Weak tail descriptors from the two unsigned tail candidates."""

    FEATURE_NAMES = [
        "tail_line_mean",
        "tail_line_max",
        "tail_extent",
        "tail_off_axis_ratio",
        "tail_near_root_ratio",
        "tail_component_ratio",
        "tail_valid",
    ]

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        sigmas=(0.5, 1.0, 1.5, 2.0),
        fan_tan: float = 2.5,
        base_half_width: float = 10.0,
        max_parallel: float = 160.0,
        near_root_px: float = 24.0,
        extractor=None,
    ):
        self.image_size = int(image_size)
        self.near_root_px = float(near_root_px)
        self._cache: Dict[str, np.ndarray] = {}
        self.extractor = (
            extractor
            if extractor is not None
            else MorphologyExtractor(
                data_dir,
                image_size=image_size,
                sigmas=sigmas,
                fan_tan=fan_tan,
                base_half_width=base_half_width,
                max_parallel=max_parallel,
                near_root_px=near_root_px,
            )
        )

    def extract(self, relpath: str) -> np.ndarray:
        relpath = str(relpath)
        cached = self._cache.get(relpath)
        if cached is not None:
            return cached.copy()

        feats = self._extract_impl(relpath)
        self._cache[relpath] = feats.astype(np.float32, copy=True)
        return feats.copy()

    def _extract_impl(self, relpath: str) -> np.ndarray:
        evidence = self.extractor.extract(relpath)
        if float(evidence["tail_valid"][0]) <= 0.0:
            return self._zero_features(valid=0.0)
        stats_neg = self._candidate_stats(
            evidence["candidate_masks"][0],
            evidence["tail_response"],
            evidence["tail_tip_neg_xy"],
            -evidence["tail_axis"],
        )
        stats_pos = self._candidate_stats(
            evidence["candidate_masks"][1],
            evidence["tail_response"],
            evidence["tail_tip_pos_xy"],
            evidence["tail_axis"],
        )

        weights = evidence["candidate_priors"].astype(np.float32)
        weights /= float(weights.sum() + 1e-8)
        stats = np.stack([stats_neg, stats_pos], axis=0)
        fused = (stats * weights[:, None]).sum(axis=0)
        fused[1] = max(float(stats_neg[1]), float(stats_pos[1]))
        fused[-1] = 1.0
        return fused.astype(np.float32)

    def _candidate_stats(
        self,
        candidate_mask: np.ndarray,
        response: np.ndarray,
        tip_xy: np.ndarray,
        axis_xy: np.ndarray,
    ) -> np.ndarray:
        cand = candidate_mask.astype(bool)
        if not np.any(cand):
            return self._zero_features(valid=0.0)

        h, w = cand.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - float(tip_xy[0])
        dy = yy - float(tip_xy[1])
        axis_xy = axis_xy.astype(np.float32)
        axis_xy = axis_xy / (np.linalg.norm(axis_xy) + 1e-8)
        perp_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float32)
        para = dx * axis_xy[0] + dy * axis_xy[1]
        off = dx * perp_xy[0] + dy * perp_xy[1]

        weighted = response * cand.astype(np.float32)
        total_w = float(weighted.sum())
        if total_w <= 1e-8:
            return self._zero_features(valid=0.0)

        region_area = float(cand.sum())
        line_mean = total_w / max(region_area, 1.0)
        line_max = float(weighted.max())

        thr = max(0.15, 0.35 * line_max)
        hit = (weighted >= thr) & cand
        extent = 0.0
        if np.any(hit):
            extent = float(
                np.clip(
                    para[hit].max() / max(self.extractor.proposer.max_parallel, 1.0),
                    0.0,
                    1.0,
                )
            )

        width = self.extractor.proposer.base_half_width + 0.35 * np.maximum(para, 0.0)
        off_axis_mask = cand & (np.abs(off) > width)
        off_axis_ratio = float(weighted[off_axis_mask].sum() / total_w)

        near_root_mask = cand & (para < self.near_root_px)
        near_root_ratio = float(weighted[near_root_mask].sum() / total_w)

        binary = hit.astype(np.uint8) * 255
        n_comp, _ = cv2.connectedComponents(binary)
        component_ratio = float(min(max(n_comp - 1, 0), 4) / 4.0)

        return np.array(
            [
                line_mean,
                line_max,
                extent,
                off_axis_ratio,
                near_root_ratio,
                component_ratio,
                1.0,
            ],
            dtype=np.float32,
        )

    def _zero_features(self, valid: float) -> np.ndarray:
        arr = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
        arr[-1] = float(valid)
        return arr


class TailRegionExtractor:
    """Tail validity/quality evidence for the morphology compiler."""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        sigmas=(0.5, 1.0, 1.5, 2.0),
        fan_tan: float = 2.5,
        base_half_width: float = 10.0,
        max_parallel: float = 160.0,
        near_root_px: float = 24.0,
        extractor=None,
    ):
        self.extractor = (
            extractor
            if extractor is not None
            else MorphologyExtractor(
                data_dir,
                image_size=image_size,
                sigmas=sigmas,
                fan_tan=fan_tan,
                base_half_width=base_half_width,
                max_parallel=max_parallel,
                near_root_px=near_root_px,
            )
        )

    def extract(self, relpath: str) -> Dict[str, np.ndarray]:
        evidence = self.extractor.extract(relpath)
        return {
            "valid": evidence["tail_valid"].astype(np.float32, copy=True),
            "tail_quality": evidence["tail_quality"].astype(np.float32, copy=True),
        }


__all__ = [
    "HeadRegionExtractor",
    "MorphologyExtractor",
    "SmidsAnatomyCache",
    "TailDescriptor",
    "TailProposal",
    "TailRegionExtractor",
    "build_empty_measurement_item",
    "build_measurement_item",
    "fit_ellipse_axis",
    "fit_mask_ellipse",
    "get_cache_paths",
    "get_smids_anatomy_cache_path",
    "head_ratio_details",
    "load_index_paths",
    "segment_head_region",
]
