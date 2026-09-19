"""Train MorphFL on SMIDS or SIPaKMeD.

Examples:
    python main.py --dataset smids --rho-enabled 1
    python main.py --dataset sipak --split-json <split.json> --rho-enabled 1
"""

import argparse
import os

from utils.Arguments import RELEASE_ROOT, build_parser, finalize_bool_args
from utils.Federation import run_experiment

_SPLITS = {
    "smids": "federated_split_v1.json",
    "sipak": "sipakmed_federated_split_v1.json",
}
_OUTPUTS = {
    "smids": "smids",
    "sipak": "sipak",
}


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", choices=["smids", "sipak"], required=True)
    known, remaining = bootstrap.parse_known_args()

    parser = build_parser()
    parser.set_defaults(
        split_json=os.path.join(RELEASE_ROOT, "data", "splits", _SPLITS[known.dataset]),
        output_dir=os.path.join(RELEASE_ROOT, "outputs", f"{_OUTPUTS[known.dataset]}_morphfl"),
    )
    args = parser.parse_args(remaining)
    finalize_bool_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
