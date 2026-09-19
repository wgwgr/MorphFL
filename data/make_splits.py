"""Generate the train / validation / test federated splits.

Two datasets are supported:

- SMIDS: each image is an independent specimen, so the split unit is the
  single image. The three clients are class-skewed non-IID partitions with
  per-class proportions (Normal/Abnormal/Non-Sperm):
      client_a 0.60 / 0.25 / 0.15
      client_b 0.25 / 0.60 / 0.15
      client_c 0.20 / 0.20 / 0.60
- SIPaKMeD: 4,049 single-cell crops originate from 966 cluster-cell images,
  so the split unit is the parent cluster image; every crop of one cluster is
  assigned to exactly one partition. The three clients receive balanced IID
  partitions (largest-cluster-first balancing).

Both datasets hold out, per class, 20% of the split units for the global test
set and 10% for the global validation set; the remaining 70% are distributed
to the clients. The random sequence (seed 42) assigns test units before
validation units, so the existing test membership is preserved exactly.

Outputs (clean/medium/hard/extreme share one partition and differ only in
client domain profiles):
    federated_split_v1.json (+ domain variants)        -- SMIDS
    sipakmed_federated_split_v1.json (+ domain variants) -- SIPaKMeD
"""

import csv
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np

SPLIT_DIR = os.path.join(REPO_ROOT, "data", "splits")
SEED = 42
TEST_RATIO = 0.20
VAL_RATIO = 0.10
CLIENTS = ["client_a", "client_b", "client_c"]

SMIDS = {
    "key": "smids",
    "data_subdir": "SMIDS",
    "classes": ["Normal_Sperm", "Abnormal_Sperm", "Non-Sperm"],
    "split_unit": "image",
    "files": {
        "clean": "federated_split_v1.json",
        "medium": "federated_split_domain_medium_v1.json",
        "hard": "federated_split_domain_hard_v1.json",
        "extreme": "federated_split_domain_extreme_v1.json",
    },
    # Per-class proportion of the training pool handed to each client.
    "client_class_ratios": {
        "Normal_Sperm": {"client_a": 0.60, "client_b": 0.25, "client_c": 0.20},
        "Abnormal_Sperm": {"client_b": 0.60, "client_a": 0.25, "client_c": 0.20},
        "Non-Sperm": {"client_c": 0.60, "client_a": 0.15, "client_b": 0.15},
    },
}
SIPAK = {
    "key": "sipakmed",
    "data_subdir": "SIPaKMeD",
    "classes": [
        "Dyskeratotic",
        "Koilocytotic",
        "Metaplastic",
        "Parabasal",
        "Superficial-Intermediate",
    ],
    "split_unit": "parent_cluster_image",
    "files": {
        "clean": "sipakmed_federated_split_v1.json",
        "medium": "sipakmed_federated_split_domain_medium_v1.json",
        "hard": "sipakmed_federated_split_domain_hard_v1.json",
        "extreme": "sipakmed_federated_split_domain_extreme_v1.json",
    },
    "client_class_ratios": None,
}


def _sipak_relpath(class_name, image_id, cell_id):
    return (
        f"images/{class_name}/im_{class_name}/CROPPED/"
        f"{int(float(image_id)):03d}_{int(float(cell_id)):02d}.bmp"
    )


def enumerate_smids():
    """Return {class: [relpath]} for every SMIDS bmp."""
    data_dir = os.path.join(REPO_ROOT, "data", SMIDS["data_subdir"])
    units = {}
    for class_name in SMIDS["classes"]:
        class_dir = os.path.join(data_dir, class_name)
        paths = sorted(
            f"{class_name}/{fname}"
            for fname in os.listdir(class_dir)
            if fname.lower().endswith(".bmp")
        )
        # One independent image per split unit (single-element group).
        units[class_name] = {path: [path] for path in paths}
    return units, os.path.abspath(data_dir)


def enumerate_sipak():
    """Return {class: {cluster_image_id: [crop relpaths]}} from the feature table."""
    data_dir = os.path.join(REPO_ROOT, "data", SIPAK["data_subdir"])
    csv_path = os.path.join(data_dir, "sipakmed_morphology_features.csv")
    clusters = defaultdict(lambda: defaultdict(list))
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clusters[row["class_name"]][row["image_id"]].append(
                _sipak_relpath(row["class_name"], row["image_id"], row["cell_id"])
            )
    units = {}
    for class_name in SIPAK["classes"]:
        for image_id in clusters[class_name]:
            clusters[class_name][image_id].sort()
        units[class_name] = dict(clusters[class_name])
    return units, os.path.abspath(data_dir)


def hold_out_units(unit_ids, rng):
    """Split one class's units into (test, val, train) id lists.

    Test units come first in the seeded permutation so test membership is
    identical to the previous test-only splits.
    """
    permuted = list(np.array(unit_ids)[rng.permutation(len(unit_ids))])
    n_test = int(round(len(permuted) * TEST_RATIO))
    n_val = int(round(len(permuted) * VAL_RATIO))
    return permuted[:n_test], permuted[n_test:n_test + n_val], permuted[n_test + n_val:]


def distribute_skewed(units_by_class, test_ids, val_ids, train_ids):
    """SMIDS: send each class's train pool to clients at class-specific ratios."""
    rng = np.random.RandomState(SEED + 1)
    clients = {c: [] for c in CLIENTS}
    for class_name in SMIDS["classes"]:
        paths = sorted(train_ids[class_name])
        rng.shuffle(paths)
        ratios = SMIDS["client_class_ratios"][class_name]
        n = len(paths)
        counts = {c: int(round(n * ratios[c])) for c in CLIENTS}
        # Absorb rounding drift into client_a so every path is assigned once.
        counts["client_a"] += n - sum(counts.values())
        start = 0
        for client in CLIENTS:
            clients[client].extend(paths[start:start + counts[client]])
            start += counts[client]
    for client in CLIENTS:
        clients[client].sort()
    return clients


def distribute_balanced(clusters_by_class, train_ids):
    """SIPaKMeD: assign whole clusters to clients, largest cluster first."""
    rng = np.random.RandomState(SEED + 1)
    clients = {c: [] for c in CLIENTS}
    client_size = {c: 0 for c in CLIENTS}
    client_class_size = {c: defaultdict(int) for c in CLIENTS}
    pending = []
    for class_name in SIPAK["classes"]:
        ids = list(train_ids[class_name])
        rng.shuffle(ids)
        for image_id in ids:
            pending.append(
                (class_name, image_id, len(clusters_by_class[class_name][image_id]))
            )
    pending.sort(key=lambda item: item[2], reverse=True)
    for class_name, image_id, n_cells in pending:
        client = min(
            CLIENTS,
            key=lambda c: (client_size[c], client_class_size[c][class_name], c),
        )
        clients[client].extend(clusters_by_class[class_name][image_id])
        client_size[client] += n_cells
        client_class_size[client][class_name] += n_cells
    for client in CLIENTS:
        clients[client].sort()
    return clients


def class_counts_smids(paths):
    counts = {c: 0 for c in SMIDS["classes"]}
    for path in paths:
        counts[path.split("/", 1)[0]] += 1
    return counts


def class_counts_sipak(paths):
    counts = {c: 0 for c in SIPAK["classes"]}
    for path in paths:
        counts[path.split("/", 2)[1]] += 1
    return counts


def cluster_counts_sipak(paths):
    seen = defaultdict(set)
    for path in paths:
        class_name = path.split("/", 2)[1]
        image_id = path.rsplit("/", 1)[-1].split("_")[0]
        seen[class_name].add(image_id)
    return {c: len(seen[c]) for c in SIPAK["classes"]}


def load_existing_profiles(filenames):
    profiles = {}
    for level, filename in filenames.items():
        path = os.path.join(SPLIT_DIR, filename)
        if os.path.exists(path):
            profiles[level] = json.load(open(path)).get("client_domain_profiles", {})
    return profiles


def build_split(dataset, units_by_class, data_dir, class_counts_fn, cluster_counts_fn=None):
    rng = np.random.RandomState(SEED)
    test_ids, val_ids, train_ids = {}, {}, {}
    for class_name in dataset["classes"]:
        test_ids[class_name], val_ids[class_name], train_ids[class_name] = hold_out_units(
            sorted(units_by_class[class_name].keys()), rng
        )

    def expand(ids_by_class):
        out = []
        for class_name in dataset["classes"]:
            for unit in ids_by_class[class_name]:
                out.extend(units_by_class[class_name][unit])
        return sorted(out)

    global_test = expand(test_ids)
    global_val = expand(val_ids)
    if dataset["split_unit"] == "parent_cluster_image":
        clients = distribute_balanced(units_by_class, train_ids)
    else:
        clients = distribute_skewed(units_by_class, test_ids, val_ids, train_ids)

    stats = {
        "global_test": class_counts_fn(global_test),
        "global_val": class_counts_fn(global_val),
        "clients": {c: class_counts_fn(paths) for c, paths in clients.items()},
        "sizes": {
            "global_test": len(global_test),
            "global_val": len(global_val),
            **{c: len(paths) for c, paths in clients.items()},
        },
    }
    if cluster_counts_fn is not None:
        stats["parent_cluster_counts"] = {
            "global_test": cluster_counts_fn(global_test),
            "global_val": cluster_counts_fn(global_val),
            "clients": {c: cluster_counts_fn(paths) for c, paths in clients.items()},
        }

    payload = {
        "dataset_name": dataset["key"],
        "data_dir": data_dir,
        "seed": SEED,
        "global_test_ratio": TEST_RATIO,
        "global_val_ratio": VAL_RATIO,
        "experiment_type": (
            "class_skewed_non_iid" if dataset["split_unit"] == "image" else "balanced_iid"
        ),
        "split_unit": dataset["split_unit"],
        "leakage_guard": "split_units_disjoint_across_clients_validation_and_test",
        "global_test_domain_eval": "clean",
        "client_domain_profiles": {},
        # The training partition is the union of the federated clients; the
        # same paths are assigned per-client in "clients".
        "train": sorted(path for paths in clients.values() for path in paths),
        "global_test": global_test,
        "global_val": global_val,
        "clients": dict(clients),
        "stats": stats,
    }
    if dataset["key"] == "smids":
        payload["client_ratios"] = {
            c: {cls: SMIDS["client_class_ratios"][cls][c] for cls in SMIDS["classes"]}
            for c in CLIENTS
        }
    else:
        payload["class_names"] = list(SIPAK["classes"])
        payload["class_to_idx"] = {c: i for i, c in enumerate(SIPAK["classes"])}
        payload["client_weights"] = {c: 1.0 for c in CLIENTS}
    return payload, test_ids, val_ids, train_ids


def write_levels(base_payload, dataset, profiles, level_meta):
    for level, filename in dataset["files"].items():
        payload = json.loads(json.dumps(base_payload))
        payload["client_domain_profiles"] = profiles.get(level, {})
        severity, requested = level_meta[level]
        payload["domain_shift_severity"] = severity
        payload["requested_domain_shift_level"] = requested
        payload["domain_shift_factor"] = ""
        with open(os.path.join(SPLIT_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  wrote {filename}")


LEVEL_META = {
    "clean": ("", ""),
    "medium": ("medium", "medium"),
    "hard": ("hard", "hard"),
    "extreme": ("extreme", "extreme"),
}


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # SMIDS
    units, data_dir = enumerate_smids()
    total = sum(len(v) for v in units.values())
    payload, test_ids, val_ids, _ = build_split(SMIDS, units, data_dir, class_counts_smids)
    print(f"SMIDS: {total} independent images; "
          f"test={payload['stats']['sizes']['global_test']} "
          f"val={payload['stats']['sizes']['global_val']} "
          f"clients={ {c: payload['stats']['sizes'][c] for c in CLIENTS} }")
    write_levels(payload, SMIDS, load_existing_profiles(SMIDS["files"]), LEVEL_META)

    # SIPaKMeD
    clusters, data_dir = enumerate_sipak()
    n_clusters = sum(len(v) for v in clusters.values())
    n_cells = sum(len(v) for ids in clusters.values() for v in ids.values())
    payload, _, _, _ = build_split(
        SIPAK, clusters, data_dir, class_counts_sipak, cluster_counts_sipak
    )
    print(f"SIPaKMeD: {n_cells} cells from {n_clusters} parent clusters; "
          f"test={payload['stats']['sizes']['global_test']} "
          f"val={payload['stats']['sizes']['global_val']} "
          f"clients={ {c: payload['stats']['sizes'][c] for c in CLIENTS} }")
    write_levels(payload, SIPAK, load_existing_profiles(SIPAK["files"]), LEVEL_META)


if __name__ == "__main__":
    main()
