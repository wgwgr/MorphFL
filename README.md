# MorphFL

Federated cell morphology classification under the morphology-basis protocol (MBP): the encoder stays private at each institution, and only declared measurement-shaped functions are shared.

## Method overview

Each client keeps a **private visual frontend** — a frozen DINOv3 ViT-B/16 encoder with small private LoRA adapters (rank 8, alpha 16) and a bounded style branch (rank-16 residual adapter, norm clamp, tanh-bounded calibrator) that absorbs residual site bias. Nothing from the encoder or the style branch ever crosses the federated boundary.

The shared surface consists of exactly three declared function classes:

- **VSP (visual semantic projection)** — maps the private CLS embedding through a shared rank-16 residual adapter into a 64-d base concept `u`, fuses `u` with a client-side measurement summary into a shared context `g`, and feeds `[h, u, g]` to the shared decision heads (a hierarchical SMIDS head: non-sperm gate + abnormality classifier; a flat head for SIPaKMeD).
- **MEM (morphometric evidence module)** — consumes only measurement-side signals (validity-masked scalars, quality/validity metadata). A compiler branch produces regional evidence concepts supervised by masked regression onto declared geometric targets; a compiled classifier emits correction logits `Δz`; a scalar-only anchor (LayerNorm + MLP) provides an imaging-perturbation-resistant prior.
- **URC (unified reliability calibration)** — two lightweight heads score per-sample path correctness: `rho_m` reads measurement-side signals only, `rho_v` reads visual-side confidence/margin signals. The log-odds difference gates the fusion, `γ = clip(sigmoid(rho_m − rho_v + b), 0, γ_max)`, blended from the heuristic conflict gate over warm-up rounds.

The final decision fuses the three paths:

```
fused = core_logits + correction_gate · Δz + anchor_gate · (anchor_logits − core_logits)
```

## Federated training protocol

Each round (`utils/Federation.py`):

1. every client restores the global shared state plus its own private state;
2. each client runs its local update (`utils/LocalTraining.py`) on its local objective — fused CE (optionally with local-prior logit adjustment and label smoothing) plus the morphology regression, the anchor CE and the URC rho BCE terms; the scalar-only anchor is pre-trained during prewarm rounds and receives extra scalar-only micro-steps;
3. the shared subset is averaged with sample-size weights (optionally with server-side momentum);
4. the personalized models are evaluated on the global validation split every round and on the test split for monitoring (frequency configurable).

Model selection uses global validation Macro-F1 (optionally EMA-smoothed); the reported score is a single global test evaluation of the best-validation-round checkpoint, so the test split never participates in selection.

## File structure

```
root/
├── main.py                  # training entry (dataset presets: smids / sipak)
├── model                    # network modules
│   ├── MorphFL.py           # full MorphFL client model (forward + fusion)
│   ├── VSP.py               # visual semantic projection
│   ├── MEM.py               # morphometric evidence module
│   ├── URC.py               # unified reliability calibration
│   ├── VisualFrontend.py    # private frozen DINOv3 + local adaptation
│   └── Layers.py            # low-rank adapter / MLP / masked loss
├── utils                    # datasets, measurement pipeline, federation logic
│   ├── Datasets.py          # SMIDS / SIPaKMeD datasets and collation
│   ├── DomainShift.py       # stain / blur / low-light profiles
│   ├── Morphology.py        # client-side measurement pipeline phi
│   ├── Federation.py        # Algorithm 1 (broadcast/local update/aggregate)
│   ├── LocalTraining.py     # one client local-update step
│   ├── BoundaryScope.py     # shared-surface declaration and aggregation
│   ├── Evaluation.py        # metrics and round logs
│   ├── Arguments.py         # command-line configuration
│   └── Utils.py             # seeding, JSON, metrics
├── audits                   # offline privacy / semantic audits
│   ├── Checkpoint.py        # checkpoint loading + method dispatch + model_config
│   ├── Baselines.py         # baseline wrappers (full fine-tune by default)
│   ├── GradientInversion.py # feature / DLG / prototype inversion
│   ├── MembershipInference.py
│   ├── BoundaryAudit.py     # per-group DLG certificate + ridge probes
│   ├── Representations.py
│   └── run_privacy_audits.py
├── data                     # datasets, splits, stain reference, split generator
│   ├── SMIDS
│   ├── SIPaKMeD
│   ├── splits
│   ├── assets
│   └── make_splits.py
├── requirements.txt
└── README.md
```

## Installation

```
pip install -r requirements.txt
```

Dependencies: torch >= 2.2, transformers >= 4.40, peft >= 0.10, numpy, pillow, opencv-python, scikit-image, scikit-learn, scipy.

The DINOv3 ViT-B/16 weights are resolved from the repository root or the enclosing workspace root directory `dinov3-vitb16-pretrain-lvd1689m/`.

## Run code

Train one job:

```
python main.py --dataset smids --rho-enabled 1
python main.py --dataset sipak --rho-enabled 1
```

`--dataset` selects the split JSON and output directory presets; pass `--split-json` / `--output-dir` to override (e.g. the domain-shift splits `federated_split_domain_{medium,hard,extreme}_v1.json`).

### Key options

Training protocol:

| Option | Default | Description |
|---|---|---|
| `--rounds` / `--local-epochs` | 20 / 2 | Federated rounds and local epochs per round |
| `--batch-size` / `--eval-batch-size` | 16 / 128 | Train / evaluation batch sizes |
| `--lr` / `--weight-decay` | 5e-4 / 0.01 | AdamW base LR (head group) and weight decay |
| `--backbone-lr-scale` / `--lora-lr-scale` | 0.2 / 0.2 | Per-group LR multipliers |
| `--shared-scope-mode` | `default` | Shared-boundary ablation: `default` (VSP+MEM+URC), `narrow` (scalar norm + anchor only), `wide` (default + private frontend) |
| `--rho-enabled` | 0 | Learn the URC reliability heads and fusion gate (0 keeps the heuristic conflict gate) |
| `--seed` / `--device` | 42 / cuda | Reproducibility and device |

Losses and scheduling (all optional features default to the paper configuration):

| Option | Default | Description |
|---|---|---|
| `--label-smoothing` | 0.0 | Label smoothing for the fused CE term only |
| `--class-balance-mode` | `none` | `logit_adjust` adds `tau * log(local class prior)` to the fused CE logits using client-local label statistics (MBP-safe) |
| `--round-lr-schedule` | `linear` | Per-round LR decay shape: `linear` or `cosine` |
| `--server-momentum-beta` | 0.0 | Server-side momentum: `global = (1-beta)*global + beta*averaged` (0.0 keeps plain FedAvg) |
| `--val-selection-ema` | 0.0 | EMA factor for the validation selection metric used by best-round choice and early stopping |

Efficiency:

| Option | Default | Description |
|---|---|---|
| `--attn-implementation` | `sdpa` | DINOv3 attention backend (mathematically equivalent to `eager`; falls back for bit-level comparison) |
| `--test-eval-every-round` | 1 | Test-evaluation frequency for monitoring; 0 disables mid-training test evaluation (the final best-val test evaluation always runs) |
| `--eval-num-workers` | -1 | Evaluation-loader workers; -1 reuses `--num-workers` |

Output artifacts (per run, under `--output-dir`):

- `args.json` — full configuration, split statistics, runtime environment, augmentation config;
- `round_logs.json` / `round_metrics.csv` — per-round train/val/test metrics and timing;
- `results.json` — summary payload (best round, test-at-best-val, server visibility, model selection metadata);
- `checkpoint_best.pth` / `checkpoint_last.pth` — best-validation and final-round states (`global_state_dict` + `client_private_states`), loadable by the audit scripts.

Generate the train/validation/test federated splits:

```
python data/make_splits.py
```

Run offline audits on a trained checkpoint:

```
python audits/run_privacy_audits.py --checkpoint-dir outputs/<run> --attacks feature grad mia cert
```

The audit output records a `model_config` block (backbone, tuning mode, LoRA rank/alpha/targets) so that every audited method's adaptation configuration is traceable. Baseline wrappers (`audits/Baselines.py`) fully fine-tune the backbone by default; the frozen+LoRA configuration is only restored for legacy checkpoints.

## Notes

- Split JSONs record a `data_dir`; if it no longer exists (e.g. after relocating the repository) `run_experiment` falls back to the bundled `<repo>/data/<dataset>` directory and prints a warning.
- The DINOv3 ViT-B/16 weights are resolved from the repository root or the enclosing workspace root directory `dinov3-vitb16-pretrain-lvd1689m/`.
- Model selection uses global validation Macro-F1; the reported score is a single global test evaluation of the best-validation-round checkpoint.
- SIPaKMeD splits are grouped by parent cluster image; SMIDS images are independent specimens. Split loading rejects any overlap between the test, validation and client partitions.
- Training augmentation is geometry-preserving (color jitter, blur, sharpness, autocontrast) because the morphology measurement pipeline requires shape fidelity; evaluation applies Reinhard stain normalization against `data/assets/stain_reference.json`.
- Per-worker numpy/random seeding makes domain-shift noise reproducible across runs.
- `result/run_table1.py` (24-job main-table grid: 3 seeds x 2 datasets x 4 shift levels) is not yet included in this release.
