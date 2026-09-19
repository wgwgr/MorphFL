"""Command-line interface for MorphFL federated training."""

import os

from utils.Utils import build_arg_parser

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_parser():
    """Define the full MorphFL command-line interface (training protocol)."""
    parser = build_arg_parser("Train MorphFL under the morphology-basis protocol")
    # Training protocol.
    parser.add_argument("--split-json", default=os.path.join(
        RELEASE_ROOT, "data", "splits", "federated_split_v1.json"))
    parser.add_argument("--output-dir", default=os.path.join(RELEASE_ROOT, "outputs", "morphfl"))
    parser.add_argument("--pretrained-model-name", default="dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--lora-lr-scale", type=float, default=0.2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--train-norm-layers", type=int, default=0,
                        help="Train normalization layers inside the private frontend.")
    parser.add_argument(
        "--shared-scope-mode",
        default="default",
        choices=["default", "narrow", "wide"],
        help=(
            "Shared-boundary width ablation: default = VSP+MEM+URC; "
            "narrow = scalar norm + anchor classifier only; "
            "wide = default boundary plus the private visual frontend."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=-1,
                        help="Workers for evaluation loaders; -1 reuses --num-workers.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-enabled", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager"],
                        help="DINOv3 attention backend; sdpa is mathematically "
                             "equivalent to eager with bf16 autocast.")
    parser.add_argument("--save-checkpoints", type=int, default=1)
    parser.add_argument("--test-eval-every-round", type=int, default=1,
                        help="Evaluate global_test every N rounds for monitoring "
                             "(0 disables mid-training test evaluation; the final "
                             "best-val test evaluation is always run).")
    parser.add_argument("--adaptive-early-stop-patience", type=int, default=0)
    parser.add_argument("--adaptive-early-stop-min-round", type=int, default=0)
    parser.add_argument("--adaptive-early-stop-min-delta", type=float, default=1e-4)
    # Architecture.
    parser.add_argument("--concept-dim", type=int, default=64)
    parser.add_argument("--morph-adapter-rank", type=int, default=16)
    parser.add_argument("--morph-adapter-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--share-global-context", type=int, default=1)
    # Private style branch.
    parser.add_argument("--style-residual-scale", type=float, default=0.12)
    parser.add_argument("--style-logit-scale", type=float, default=0.05)
    parser.add_argument("--style-delta-max-norm", type=float, default=0.60)
    # MEM compiler.
    parser.add_argument("--compiler-input-dim", type=int, default=6)
    parser.add_argument("--compiler-hidden-dim", type=int, default=128)
    parser.add_argument("--compiler-supervision-weight", type=float, default=0.25)
    parser.add_argument("--content-conflict-decay", type=float, default=0.30)
    parser.add_argument("--content-conflict-anchor-strength", type=float, default=0.85)
    parser.add_argument("--compiler-conflict-decay", type=float, default=0.75)
    parser.add_argument("--compiler-conflict-anchor-strength", type=float, default=0.85)
    # Scalar anchor and heuristic gate (URC warm-up teacher and URC-off fallback).
    parser.add_argument("--anchor-floor", type=float, default=0.0)
    parser.add_argument("--anchor-scale", type=float, default=0.7)
    parser.add_argument("--anchor-conflict-boost", type=float, default=0.80)
    parser.add_argument("--anchor-max", type=float, default=0.95)
    parser.add_argument("--anchor-takeover-mid", type=float, default=0.45)
    parser.add_argument("--anchor-takeover-width", type=float, default=0.10)
    parser.add_argument("--anchor-supervision-weight", type=float, default=2.0)
    parser.add_argument("--anchor-micro-steps", type=int, default=4)
    parser.add_argument("--anchor-prewarm-rounds", type=int, default=2)
    parser.add_argument("--anchor-prewarm-micro-steps", type=int, default=6)
    # Losses and per-round LR schedule.
    parser.add_argument("--base-loss-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for the fused CE term only "
                             "(anchor and rho losses are untouched).")
    parser.add_argument("--class-balance-mode", default="none",
                        choices=["none", "logit_adjust"],
                        help="Class-imbalance handling for the fused CE term: "
                             "logit_adjust adds tau * log(local class prior) to "
                             "the logits using client-local label statistics.")
    parser.add_argument("--class-balance-tau", type=float, default=1.0,
                        help="Temperature of the logit-adjustment correction.")
    parser.add_argument("--round-lr-schedule", default="linear",
                        choices=["linear", "cosine"],
                        help="Per-round learning-rate decay shape.")
    parser.add_argument("--round-lr-scale-start", type=float, default=1.0)
    parser.add_argument("--round-lr-scale-end", type=float, default=0.50)
    parser.add_argument("--round-lr-decay-start-round", type=int, default=6)
    parser.add_argument("--server-momentum-beta", type=float, default=0.0,
                        help="Server-side momentum for the averaged shared state: "
                             "global = (1 - beta) * global + beta * averaged.")
    parser.add_argument("--val-selection-ema", type=float, default=0.0,
                        help="EMA factor for the validation selection metric "
                             "(0 disables; e.g. 0.3 smooths best-round choice and "
                             "early stopping).")
    # URC reliability heads.
    parser.add_argument("--rho-enabled", type=int, default=0,
                        help="Learn rho_m/rho_v reliability heads and their fusion gate "
                             "(0 keeps the heuristic conflict gate).")
    parser.add_argument("--rho-warmup-rounds", type=int, default=4,
                        help="Rounds over which the heuristic gate blends into the learned gate.")
    parser.add_argument("--rho-loss-weight", type=float, default=0.15,
                        help="BCE weight for the reliability heads.")
    parser.add_argument("--rho-hidden-dim", type=int, default=32)
    return parser


def finalize_bool_args(args) -> None:
    """Normalize int-typed boolean flags and pin the method identifier."""
    args.method = "morphfl"
    args.train_norm_layers = bool(args.train_norm_layers)
    args.share_global_context = bool(args.share_global_context)
    args.amp_enabled = bool(args.amp_enabled)
    args.rho_enabled = bool(args.rho_enabled)
    args.save_checkpoints = bool(args.save_checkpoints)


def main():
    """CLI entry point when running utils/Arguments.py directly."""
    from utils.Federation import run_experiment

    args = build_parser().parse_args()
    finalize_bool_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
