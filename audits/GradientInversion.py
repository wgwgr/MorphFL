"""Feature inversion, DLG gradient inversion and prototype inversion."""


import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audits.Representations import collect_representations, exp_forward


class Decoder224(nn.Module):
    """Transpose-convolution decoder from a representation vector to 224x224."""

    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.15):
        super().__init__()
        hidden_dim = hidden_dim or min(256, max(32, in_dim))
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8 * 7 * 7),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(8, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.project(z).view(-1, 8, 7, 7)
        return self.up(hidden)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.detach(), target.detach())
    if mse.item() < 1e-12:
        return 100.0
    return -10.0 * math.log10(mse.item())


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, size, 1) * g.view(1, 1, 1, size)


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1, img2 = img1.detach(), img2.detach()
    channels = int(img1.shape[1])
    kernel = _gaussian_kernel(11, 1.5).expand(channels, 1, -1, -1).to(img1.device)
    mu1 = F.conv2d(img1, kernel, padding=5, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=5, groups=channels)
    s1 = F.conv2d(img1 * img1, kernel, padding=5, groups=channels) - mu1 * mu1
    s2 = F.conv2d(img2 * img2, kernel, padding=5, groups=channels) - mu2 * mu2
    s12 = F.conv2d(img1 * img2, kernel, padding=5, groups=channels) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
    )
    return float(ssim_map.mean().item())


def feature_inversion(
    exp: Dict,
    output_dir: str,
    train_ratio: float = 0.7,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_samples: int = 1024,
    patience: int = 30,
) -> Dict:
    device = exp["device"]
    collected = collect_representations(exp, max_samples=max_samples)
    images = collected["images"]
    n = images.size(0)
    n_train = max(1, int(n * train_ratio))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    repr_keys = [
        ("private_feature", "from_private_feature"),
        ("base_concept", "from_vsp_base_concept"),
        ("global_context", "from_vsp_context"),
        ("compiled_head_concept", "from_mem_compiled_head"),
        ("compiled_tail_concept", "from_mem_compiled_tail"),
        ("scalar", "from_scalar_measurements"),
    ]
    results: Dict[str, Dict] = {}
    os.makedirs(output_dir, exist_ok=True)
    for key, name in repr_keys:
        feats = collected[key]
        if feats.numel() == 0 or feats.dim() < 2:
            results[name] = {"note": "empty", "skipped": True}
            continue
        in_dim = feats.size(1)
        decoder = Decoder224(in_dim).to(device)
        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=lr, weight_decay=5e-4 if in_dim >= 256 else 1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        x_train, y_train = feats[train_idx].to(device), images[train_idx].to(device)
        x_test, y_test = feats[test_idx].to(device), images[test_idx].to(device)
        best_psnr, best_ssim, best_l1, best_state = None, None, None, None
        no_improve = 0
        for epoch in range(epochs):
            decoder.train()
            order = torch.randperm(x_train.size(0), generator=torch.Generator().manual_seed(epoch))
            for start in range(0, x_train.size(0), batch_size):
                batch_idx = order[start:start + batch_size]
                pred = decoder(x_train[batch_idx])
                loss = F.l1_loss(pred, y_train[batch_idx]) + 0.5 * F.mse_loss(
                    pred, y_train[batch_idx]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()
            decoder.eval()
            with torch.inference_mode():
                preds = decoder(x_test)
                l1 = F.l1_loss(preds, y_test).item()
                psnr_v = psnr(preds, y_test)
                ssim_v = ssim(preds, y_test)
            if best_psnr is None or psnr_v > best_psnr:
                best_psnr, best_ssim, best_l1 = psnr_v, ssim_v, l1
                best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        results[name] = {
            "repr_dim": in_dim,
            "n_train": int(n_train),
            "n_test": int(n - n_train),
            "epochs_run": epoch + 1,
            "best_test_l1": best_l1,
            "best_psnr_db": best_psnr,
            "best_ssim": best_ssim,
        }
        if best_state is not None:
            torch.save(best_state, os.path.join(output_dir, f"decoder_{name}.pth"))
    return results


def _named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def gradient_inversion_dlg(
    exp: Dict,
    output_dir: str,
    n_trials: int = 4,
    iterations: int = 2000,
    lr: float = 0.1,
    n_restarts: int = 3,
    tv_weight: float = 1e-3,
    shared_key_filter: Optional[List[str]] = None,
    setups: Optional[List[Tuple[str, bool]]] = None,
) -> Dict:
    device = exp["device"]
    model = exp["global_model"]
    shared_prefixes = list(exp["shared_keys"])
    if shared_key_filter:
        shared_prefixes = [k for k in shared_prefixes if k.startswith(tuple(shared_key_filter))]
    if setups is None:
        setups = [("full_params", False), ("shared_params_only", True)]

    samples = []
    for batch in exp["test_loader"]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        for j in range(min(n_trials, images.size(0))):
            samples.append({
                "image": images[j:j + 1].clone(),
                "label": labels[j:j + 1].clone(),
                "scalar": batch["scalar_feats"][j:j + 1].to(device),
                "valid": batch["validity_mask"][j:j + 1].to(device),
                "relpath": [batch["relpath"][j]],
            })
        if len(samples) >= n_trials:
            break

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict] = {}
    for setup_name, use_shared_only in setups:
        setup_psnr, setup_ssim, setup_l1 = [], [], []
        for trial, sample in enumerate(samples):
            gt_image, gt_label = sample["image"], sample["label"]
            reference_batch = {
                "image": gt_image.clone(),
                "label": gt_label.clone(),
                "scalar_feats": sample["scalar"].clone(),
                "validity_mask": sample["valid"].clone(),
                "relpath": sample["relpath"],
            }
            model.eval()
            for _, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.zero_()
            with torch.enable_grad():
                out, labels = exp_forward(exp, reference_batch, use_amp=False)
                loss = F.cross_entropy(out["fusion_logits"].float(), labels)
            named_params = _named_parameters(model)
            reference_grads = torch.autograd.grad(
                loss, [p for _, p in named_params], allow_unused=True
            )
            if use_shared_only:
                select = []
                for (name, param), grad in zip(named_params, reference_grads):
                    if any(name.startswith(prefix) for prefix in shared_prefixes):
                        select.append((
                            name,
                            param,
                            grad.detach().clone() if grad is not None else torch.zeros_like(param.data),
                        ))
            else:
                select = [
                    (name, param, grad.detach().clone() if grad is not None else torch.zeros_like(param.data))
                    for (name, param), grad in zip(named_params, reference_grads)
                ]
            target_params = [item[1] for item in select]
            target_grads = [item[2].to(device) for item in select]
            n_target = sum(grad.numel() for grad in target_grads)

            best_psnr, best_ssim, best_l1, best_dummy = -1.0, -1.0, 1e9, None
            for restart in range(n_restarts):
                torch.manual_seed(42 + restart * 137 + trial * 7)
                dummy = torch.randn_like(gt_image, device=device, requires_grad=True)
                optimizer = torch.optim.Adam([dummy], lr=lr, betas=(0.9, 0.999))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=iterations, eta_min=lr * 0.01
                )
                for _ in range(iterations):
                    optimizer.zero_grad(set_to_none=True)
                    for param in target_params:
                        if param.grad is not None:
                            param.grad.zero_()
                    with torch.enable_grad():
                        fake_batch = {
                            "image": dummy.clamp(0.0, 1.0),
                            "label": gt_label,
                            "scalar_feats": sample["scalar"],
                            "validity_mask": sample["valid"],
                            "relpath": sample["relpath"],
                        }
                        model.eval()
                        fake_out, fake_labels = exp_forward(exp, fake_batch, use_amp=False)
                        fake_loss = F.cross_entropy(
                            fake_out["fusion_logits"].float(), fake_labels
                        )
                        fake_grads = torch.autograd.grad(
                            fake_loss, target_params, create_graph=True, allow_unused=True
                        )
                    diff_parts = []
                    for fake_grad, target_grad in zip(fake_grads, target_grads):
                        if fake_grad is None:
                            continue
                        diff_parts.append(
                            (fake_grad.reshape(-1) - target_grad.reshape(-1)).pow(2).sum()
                            / max(1, target_grad.numel())
                        )
                    diff = torch.stack(diff_parts).mean() if diff_parts else torch.tensor(
                        0.0, device=device, requires_grad=True
                    )
                    with torch.no_grad():
                        tv_x = torch.abs(dummy[:, :, 1:, :] - dummy[:, :, :-1, :]).mean()
                        tv_y = torch.abs(dummy[:, :, :, 1:] - dummy[:, :, :, :-1]).mean()
                    total = diff + tv_weight * (tv_x + tv_y) * 0.5
                    total.backward()
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        dummy.clamp_(0.0, 1.0)
                        pred = dummy.detach().clamp(0.0, 1.0)
                        psnr_v, ssim_v = psnr(pred, gt_image), ssim(pred, gt_image)
                        l1_v = F.l1_loss(pred, gt_image).item()
                    if psnr_v > best_psnr:
                        best_psnr, best_ssim, best_l1 = psnr_v, ssim_v, l1_v
                        best_dummy = dummy.detach().clone().clamp(0.0, 1.0)
            setup_psnr.append(best_psnr)
            setup_ssim.append(best_ssim)
            setup_l1.append(best_l1)
            if trial == 0 and best_dummy is not None:
                torch.save({
                    "ground_truth": gt_image.detach().cpu(),
                    "reconstruction": best_dummy.cpu(),
                    "label": int(gt_label.item()),
                }, os.path.join(output_dir, f"inversion_{setup_name}_sample{trial}.pth"))
        results[setup_name] = {
            "n_trials": len(samples),
            "n_restarts": n_restarts,
            "iterations": iterations,
            "n_target_params": n_target,
            "shared_only": use_shared_only,
            "mean_psnr_db": float(np.mean(setup_psnr)) if setup_psnr else None,
            "std_psnr_db": float(np.std(setup_psnr)) if len(setup_psnr) > 1 else 0.0,
            "mean_ssim": float(np.mean(setup_ssim)) if setup_ssim else None,
            "std_ssim": float(np.std(setup_ssim)) if len(setup_ssim) > 1 else 0.0,
            "mean_l1": float(np.mean(setup_l1)) if setup_l1 else None,
            "per_trial_psnr": setup_psnr,
            "per_trial_ssim": setup_ssim,
            "per_trial_l1": setup_l1,
        }
    return results


def prototype_inversion(
    exp: Dict,
    output_dir: str,
    iterations: int = 1000,
    lr: float = 0.1,
    n_restarts: int = 2,
    tv_weight: float = 1e-3,
) -> Dict:
    """Invert FedProto class prototypes against class-mean test images."""
    device = exp["device"]
    model = exp["global_model"]
    prototypes = exp.get("fedproto_prototypes")
    if prototypes is None:
        return {"error": "no fedproto_prototypes in experiment"}

    class_images: Dict[int, List[torch.Tensor]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in exp["test_loader"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            for j in range(images.size(0)):
                class_images.setdefault(int(labels[j].item()), []).append(images[j].cpu())
    class_mean_images = {}
    for class_idx in range(prototypes.size(0)):
        if class_images.get(class_idx):
            class_mean_images[class_idx] = torch.stack(class_images[class_idx]).mean(dim=0).to(device)

    os.makedirs(output_dir, exist_ok=True)
    per_class_psnr, per_class_ssim = [], []
    for class_idx in range(prototypes.size(0)):
        target = prototypes[class_idx:class_idx + 1].to(device)
        gt_image = class_mean_images.get(class_idx)
        if gt_image is None:
            per_class_psnr.append(-1.0)
            per_class_ssim.append(0.0)
            continue
        gt_image = gt_image.unsqueeze(0)
        best_psnr, best_ssim, best_dummy = -1.0, -1.0, None
        for restart in range(n_restarts):
            torch.manual_seed(42 + restart * 137 + class_idx * 7)
            dummy = torch.randn_like(gt_image, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([dummy], lr=lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=iterations, eta_min=lr * 0.01
            )
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    out = model(dummy.clamp(0.0, 1.0))
                    feature_loss = F.mse_loss(out["pooled"], target)
                with torch.no_grad():
                    tv_x = torch.abs(dummy[:, :, 1:, :] - dummy[:, :, :-1, :]).mean()
                    tv_y = torch.abs(dummy[:, :, :, 1:] - dummy[:, :, :, :-1]).mean()
                (feature_loss + tv_weight * (tv_x + tv_y) * 0.5).backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    dummy.clamp_(0.0, 1.0)
            with torch.no_grad():
                pred = dummy.detach().clamp(0.0, 1.0)
                psnr_v, ssim_v = psnr(pred, gt_image), ssim(pred, gt_image)
            if psnr_v > best_psnr:
                best_psnr, best_ssim, best_dummy = psnr_v, ssim_v, pred.cpu().clone()
        per_class_psnr.append(best_psnr)
        per_class_ssim.append(best_ssim)
        if best_dummy is not None:
            torch.save({
                "ground_truth_class_mean": gt_image.cpu(),
                "reconstruction": best_dummy,
                "class_idx": class_idx,
                "target_prototype": target.cpu(),
            }, os.path.join(output_dir, f"prototype_class{class_idx}.pth"))

    valid_psnr = [v for v in per_class_psnr if v > -1.0]
    valid_ssim = [v for v in per_class_ssim if v > -1.0]
    return {
        "n_classes": int(prototypes.size(0)),
        "iterations": iterations,
        "n_restarts": n_restarts,
        "mean_psnr_db": float(np.mean(valid_psnr)) if valid_psnr else None,
        "std_psnr_db": float(np.std(valid_psnr)) if len(valid_psnr) > 1 else 0.0,
        "mean_ssim": float(np.mean(valid_ssim)) if valid_ssim else None,
        "per_class_psnr": per_class_psnr,
        "per_class_ssim": per_class_ssim,
        "shared_only": True,
    }
