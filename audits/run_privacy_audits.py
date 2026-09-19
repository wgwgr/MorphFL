"""Offline privacy and semantic audits of a trained checkpoint.

Examples:
    python audits/run_privacy_audits.py \\
        --checkpoint-dir outputs/smids_extreme_seed42 \\
        --attacks feature grad mia cert

The method (morphfl / fedavg / fedbn / fedprox / ditto / fedproto /
local_only) is read from the checkpoint arguments.
"""


import argparse
import os
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# audits/ also needs repo root (set above) to import flat modules.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from audits.Checkpoint import prepare_experiment
from audits.GradientInversion import feature_inversion, gradient_inversion_dlg, prototype_inversion
from audits.MembershipInference import membership_inference
from audits.BoundaryAudit import boundary_certificate
from utils.Utils import set_seed, write_json


def main():
    parser = argparse.ArgumentParser("run_privacy_audits")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Run directory containing checkpoint_last.pth.")
    parser.add_argument("--attacks", nargs="+", default=["feature", "grad", "mia"],
                        choices=["feature", "grad", "mia", "cert"])
    parser.add_argument("--cert-groups", default="",
                        help="Comma-separated boundary group names (default: all present).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--dlg-iterations", type=int, default=2000)
    parser.add_argument("--dlg-restarts", type=int, default=3)
    parser.add_argument("--dlg-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    exp = prepare_experiment(
        args.checkpoint_dir, device, eval_batch_size=96, num_workers=args.num_workers
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(REPO_ROOT, "outputs", f"privacy_audits_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_path": exp["ckpt_path"],
        "method": exp["args_saved"].get("method", "morphfl"),
        "is_smids": exp["is_smids"],
        "best_macro_f1": exp["summary"].get("best_macro_f1"),
        "model_config": exp.get("model_config", {}),
        "attacks_requested": list(args.attacks),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(os.path.join(out_dir, "experiment_meta.json"), meta)
    results = {"meta": meta}

    if "feature" in args.attacks:
        print("[1/4] feature inversion ...")
        t0 = time.perf_counter()
        feature_result = feature_inversion(
            exp, os.path.join(out_dir, "feature_inversion"), max_samples=args.max_samples
        )
        results["feature_inversion"] = feature_result
        print(f"[1/4] done in {time.perf_counter() - t0:.1f}s")

    if "grad" in args.attacks:
        method = exp["args_saved"].get("method", "morphfl")
        if method == "fedproto":
            print("[2/4] prototype inversion ...")
            grad_result = {
                "shared_params_only": prototype_inversion(
                    exp,
                    os.path.join(out_dir, "prototype_inversion"),
                    iterations=args.dlg_iterations,
                    n_restarts=args.dlg_restarts,
                    lr=args.dlg_lr,
                )
            }
        elif method == "local_only":
            print("[2/4] gradient inversion skipped: nothing is shared")
            grad_result = {
                "shared_params_only": {
                    "mean_psnr_db": None,
                    "mean_ssim": None,
                    "note": "Local-only exchanges no parameters or gradients",
                    "n_trials": 0,
                    "shared_only": True,
                }
            }
        else:
            print("[2/4] DLG gradient inversion ...")
            grad_result = gradient_inversion_dlg(
                exp,
                os.path.join(out_dir, "gradient_inversion"),
                iterations=args.dlg_iterations,
                n_restarts=args.dlg_restarts,
                lr=args.dlg_lr,
            )
        results["gradient_inversion_dlg"] = grad_result

    if "mia" in args.attacks:
        print("[3/4] membership inference ...")
        t0 = time.perf_counter()
        results["membership_inference"] = membership_inference(exp)
        print(f"[3/4] done in {time.perf_counter() - t0:.1f}s")

    if "cert" in args.attacks:
        print("[4/4] boundary certificate ...")
        t0 = time.perf_counter()
        group_filter = [s.strip() for s in args.cert_groups.split(",") if s.strip()] or None
        results["boundary_certificate"] = boundary_certificate(
            exp,
            os.path.join(out_dir, "boundary_certificate"),
            dlg_iterations=max(400, args.dlg_iterations // 3),
            dlg_restarts=2,
            full_iterations=args.dlg_iterations,
            full_restarts=args.dlg_restarts,
            group_filter=group_filter,
        )
        print(f"[4/4] done in {time.perf_counter() - t0:.1f}s")

    results["meta"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(os.path.join(out_dir, "attack_results.json"), results)
    print(f"results saved to: {out_dir}/attack_results.json")


if __name__ == "__main__":
    main()
