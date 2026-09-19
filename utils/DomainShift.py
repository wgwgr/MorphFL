"""Persistent site-level domain-shift simulation.

Three client imaging profiles (stain, blur, low-light) are applied at four
severity levels (clean, medium, hard, extreme) through a deterministic
PIL/OpenCV pipeline. The same profile is applied to a client's local
training split and to the held-out global test split, implementing the
matched train/test protocol of the paper.

The ``name`` field of each profile identifies the parameter set version and
is used to key on-disk measurement caches; it must be changed whenever the
profile parameters change.
"""


import json
from typing import Dict, Mapping, Optional

import cv2
import numpy as np
from PIL import Image


_PRESET_PROFILES: Dict[str, Dict[str, object]] = {
    "clean": {
        "name": "clean",
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "rgb_gains": [1.0, 1.0, 1.0],
        "gamma": 1.0,
        "grayscale_mix": 0.0,
        "noise_std": 0.0,
        "blur_sigma": 0.0,
        "downsample_scale": 1.0,
    },
    "stain_mild": {
        "name": "stain_mild",
        "brightness": 1.04,
        "contrast": 0.97,
        "saturation": 0.93,
        "rgb_gains": [1.10, 1.00, 0.94],
        "gamma": 1.0,
        "grayscale_mix": 0.0,
        "noise_std": 0.0,
        "blur_sigma": 0.0,
        "downsample_scale": 1.0,
    },
    "stain_medium": {
        "name": "stain_medium_v2",
        "brightness": 1.10,
        "contrast": 0.90,
        "saturation": 0.78,
        "rgb_gains": [1.20, 0.99, 0.82],
        "gamma": 1.0,
        "grayscale_mix": 0.05,
        "noise_std": 2.5,
        "blur_sigma": 0.35,
        "downsample_scale": 0.91,
    },
    "stain_severe": {
        "name": "stain_severe_v2",
        "brightness": 1.15,
        "contrast": 0.84,
        "saturation": 0.58,
        "rgb_gains": [1.30, 0.96, 0.68],
        "gamma": 1.05,
        "grayscale_mix": 0.18,
        "noise_std": 5.0,
        "blur_sigma": 0.7,
        "downsample_scale": 0.84,
    },
    "stain_extreme": {
        "name": "stain_extreme_v3",
        "brightness": 1.18,
        "contrast": 0.76,
        "saturation": 0.46,
        "rgb_gains": [1.42, 0.91, 0.58],
        "gamma": 1.10,
        "grayscale_mix": 0.14,
        "noise_std": 4.5,
        "blur_sigma": 0.9,
        "downsample_scale": 0.80,
    },
    "blur_mild": {
        "name": "blur_mild",
        "brightness": 0.98,
        "contrast": 0.95,
        "saturation": 1.0,
        "rgb_gains": [1.0, 1.0, 1.0],
        "gamma": 1.0,
        "grayscale_mix": 0.0,
        "noise_std": 0.0,
        "blur_sigma": 0.8,
        "downsample_scale": 0.85,
    },
    "blur_medium": {
        "name": "blur_medium_v2",
        "brightness": 0.95,
        "contrast": 0.86,
        "saturation": 0.90,
        "rgb_gains": [1.0, 1.0, 1.0],
        "gamma": 1.0,
        "grayscale_mix": 0.03,
        "noise_std": 2.0,
        "blur_sigma": 1.8,
        "downsample_scale": 0.62,
    },
    "blur_severe": {
        "name": "blur_severe_v2",
        "brightness": 0.92,
        "contrast": 0.78,
        "saturation": 0.82,
        "rgb_gains": [1.0, 1.0, 1.0],
        "gamma": 1.05,
        "grayscale_mix": 0.10,
        "noise_std": 4.5,
        "blur_sigma": 2.6,
        "downsample_scale": 0.50,
    },
    "blur_extreme": {
        "name": "blur_extreme_v3",
        "brightness": 0.90,
        "contrast": 0.70,
        "saturation": 0.74,
        "rgb_gains": [1.0, 1.0, 1.0],
        "gamma": 1.08,
        "grayscale_mix": 0.08,
        "noise_std": 4.0,
        "blur_sigma": 3.4,
        "downsample_scale": 0.42,
    },
    "lowlight_mild": {
        "name": "lowlight_mild",
        "brightness": 0.92,
        "contrast": 1.02,
        "saturation": 0.96,
        "rgb_gains": [0.96, 0.98, 1.05],
        "gamma": 1.0,
        "grayscale_mix": 0.0,
        "noise_std": 0.0,
        "blur_sigma": 0.2,
        "downsample_scale": 1.0,
    },
    "lowlight_medium": {
        "name": "lowlight_medium_v2",
        "brightness": 0.78,
        "contrast": 1.12,
        "saturation": 0.78,
        "rgb_gains": [0.90, 0.95, 1.06],
        "gamma": 1.18,
        "grayscale_mix": 0.08,
        "noise_std": 4.0,
        "blur_sigma": 0.7,
        "downsample_scale": 0.88,
    },
    "lowlight_severe": {
        "name": "lowlight_severe_v2",
        "brightness": 0.60,
        "contrast": 1.16,
        "saturation": 0.42,
        "rgb_gains": [0.86, 0.92, 1.04],
        "gamma": 1.35,
        "grayscale_mix": 0.35,
        "noise_std": 7.0,
        "blur_sigma": 1.0,
        "downsample_scale": 0.78,
    },
    "lowlight_extreme": {
        "name": "lowlight_extreme_v3",
        "brightness": 0.52,
        "contrast": 0.88,
        "saturation": 0.34,
        "rgb_gains": [0.84, 0.90, 1.08],
        "gamma": 1.42,
        "grayscale_mix": 0.26,
        "noise_std": 9.0,
        "blur_sigma": 1.35,
        "downsample_scale": 0.70,
    },
}


def canonicalize_domain_profile(profile: Optional[Mapping[str, object] | str]) -> Dict[str, object]:
    if profile is None:
        return dict(_PRESET_PROFILES["clean"])
    if isinstance(profile, str):
        preset = _PRESET_PROFILES.get(profile)
        if preset is None:
            raise KeyError(f"Unknown domain shift preset: {profile}")
        return dict(preset)
    out = dict(_PRESET_PROFILES["clean"])
    out.update(dict(profile))
    if "name" not in out:
        out["name"] = "custom"
    rgb_gains = out.get("rgb_gains", [1.0, 1.0, 1.0])
    if len(rgb_gains) != 3:
        raise ValueError("domain shift rgb_gains must have length 3")
    out["rgb_gains"] = [float(x) for x in rgb_gains]
    for key in (
        "brightness",
        "contrast",
        "saturation",
        "gamma",
        "grayscale_mix",
        "noise_std",
        "blur_sigma",
        "downsample_scale",
    ):
        out[key] = float(out.get(key, _PRESET_PROFILES["clean"][key]))
    return out


def has_domain_shift(profile: Optional[Mapping[str, object] | str]) -> bool:
    prof = canonicalize_domain_profile(profile)
    return domain_profile_cache_key(prof) != domain_profile_cache_key("clean")


def domain_profile_cache_key(profile: Optional[Mapping[str, object] | str]) -> str:
    prof = canonicalize_domain_profile(profile)
    return json.dumps(prof, sort_keys=True, ensure_ascii=True)


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return arr
    ksize = max(3, int(round(float(sigma) * 4.0)) * 2 + 1)
    return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def _down_up_sample(arr: np.ndarray, scale: float) -> np.ndarray:
    scale = float(scale)
    if scale >= 0.999:
        return arr
    h, w = arr.shape[:2]
    small_w = max(16, int(round(w * scale)))
    small_h = max(16, int(round(h * scale)))
    down = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


def _adjust_saturation_rgb(arr_rgb_uint8: np.ndarray, saturation: float) -> np.ndarray:
    if abs(float(saturation) - 1.0) < 1e-6:
        return arr_rgb_uint8
    hsv = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= float(saturation)
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _adjust_gamma_rgb(arr_rgb_uint8: np.ndarray, gamma: float) -> np.ndarray:
    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-6:
        return arr_rgb_uint8
    work = np.clip(arr_rgb_uint8.astype(np.float32) / 255.0, 0.0, 1.0)
    work = np.power(work, gamma)
    return _clip_uint8(work * 255.0)


def _mix_grayscale_rgb(arr_rgb_uint8: np.ndarray, grayscale_mix: float) -> np.ndarray:
    grayscale_mix = float(grayscale_mix)
    if grayscale_mix <= 1e-6:
        return arr_rgb_uint8
    gray = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    work = (1.0 - grayscale_mix) * arr_rgb_uint8.astype(np.float32) + grayscale_mix * gray_rgb
    return _clip_uint8(work)


def _add_gaussian_noise_rgb(arr_rgb_uint8: np.ndarray, noise_std: float) -> np.ndarray:
    noise_std = float(noise_std)
    if noise_std <= 1e-6:
        return arr_rgb_uint8
    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr_rgb_uint8.shape).astype(np.float32)
    work = arr_rgb_uint8.astype(np.float32) + noise
    return _clip_uint8(work)


def apply_domain_shift_rgb(rgb: np.ndarray, profile: Optional[Mapping[str, object] | str]) -> np.ndarray:
    prof = canonicalize_domain_profile(profile)
    arr = np.asarray(rgb, dtype=np.uint8)
    arr = _down_up_sample(arr, prof["downsample_scale"])
    arr = _gaussian_blur(arr, prof["blur_sigma"])

    work = arr.astype(np.float32)
    work *= float(prof["brightness"])
    work = (work - 127.5) * float(prof["contrast"]) + 127.5
    work *= np.asarray(prof["rgb_gains"], dtype=np.float32).reshape(1, 1, 3)
    arr = _clip_uint8(work)
    arr = _adjust_saturation_rgb(arr, prof["saturation"])
    arr = _adjust_gamma_rgb(arr, prof["gamma"])
    arr = _mix_grayscale_rgb(arr, prof["grayscale_mix"])
    arr = _add_gaussian_noise_rgb(arr, prof["noise_std"])
    return arr


def apply_domain_shift_pil(image: Image.Image, profile: Optional[Mapping[str, object] | str]) -> Image.Image:
    prof = canonicalize_domain_profile(profile)
    if not has_domain_shift(prof):
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    shifted = apply_domain_shift_rgb(rgb, prof)
    return Image.fromarray(shifted, mode="RGB")


def apply_domain_shift_bgr(bgr: np.ndarray, profile: Optional[Mapping[str, object] | str]) -> np.ndarray:
    prof = canonicalize_domain_profile(profile)
    if not has_domain_shift(prof):
        return bgr
    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    shifted = apply_domain_shift_rgb(rgb, prof)
    return cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR)


__all__ = [
    "apply_domain_shift_bgr",
    "apply_domain_shift_pil",
    "canonicalize_domain_profile",
    "domain_profile_cache_key",
    "has_domain_shift",
]
