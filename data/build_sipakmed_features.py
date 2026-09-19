#!/usr/bin/env python3
"""Build a unified SIPaKMeD morphology feature table from official feature files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASS_SPECS: Sequence[Tuple[str, str]] = (
    ("DYSKERATOTIC", "Dyskeratotic"),
    ("KOILOCYTOTIC", "Koilocytotic"),
    ("METAPLASTIC", "Metaplastic"),
    ("PARABASAL", "Parabasal"),
    ("SUP_INT", "Superficial-Intermediate"),
)

SHAPE_FEATURES: Sequence[str] = (
    "area",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "orientation_deg",
    "equivalent_diameter",
    "solidity",
    "extent",
)

TEXTURE_STATS: Sequence[str] = (
    "avg_intensity",
    "avg_contrast",
    "smoothness",
    "uniformity",
    "third_moment",
    "entropy",
)

CHANNELS: Sequence[str] = ("r", "g", "b")


def build_feature_names(prefix: str) -> List[str]:
    names = [f"{prefix}_{name}" for name in SHAPE_FEATURES]
    for channel in CHANNELS:
        for stat in TEXTURE_STATS:
            names.append(f"{prefix}_{channel}_{stat}")
    return names


CYTOPLASM_FEATURE_NAMES = build_feature_names("cytoplasm")
NUCLEUS_FEATURE_NAMES = build_feature_names("nucleus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge official SIPaKMeD nucleus/cytoplasm feature tables and derive morphology ratios."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/SIPaKMeD",
        help="Path to the SIPaKMeD directory containing Features_CELL (default: data/SIPaKMeD).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/SIPaKMeD/sipakmed_morphology_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--schema-md",
        default="data/SIPaKMeD/sipakmed_morphology_features_schema.md",
        help="Output Markdown schema path.",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def normalize_orientation_gap(a: float, b: float) -> float:
    gap = abs(a - b) % 180.0
    if gap > 90.0:
        gap = 180.0 - gap
    return gap


def iter_rows(path: Path) -> Iterable[List[str]]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                yield row


def parse_region_rows(path: Path, feature_names: Sequence[str]) -> Dict[Tuple[int, int], Dict[str, float]]:
    records: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in iter_rows(path):
        if len(row) != 28:
            raise ValueError(f"{path} has malformed row with {len(row)} columns, expected 28.")
        image_id = int(float(row[0]))
        cell_id = int(float(row[1]))
        values = [float(x) for x in row[2:]]
        record = {name: value for name, value in zip(feature_names, values)}
        records[(image_id, cell_id)] = record
    return records


def derive_features(record: Dict[str, float]) -> Dict[str, float]:
    cyt_area = record["cytoplasm_area"]
    nuc_area = record["nucleus_area"]
    cyt_major = record["cytoplasm_major_axis_length"]
    nuc_major = record["nucleus_major_axis_length"]
    cyt_minor = record["cytoplasm_minor_axis_length"]
    nuc_minor = record["nucleus_minor_axis_length"]
    cyt_eq = record["cytoplasm_equivalent_diameter"]
    nuc_eq = record["nucleus_equivalent_diameter"]

    derived = {
        "nc_area_ratio": safe_div(nuc_area, cyt_area),
        "cytoplasm_to_nucleus_area_ratio": safe_div(cyt_area, nuc_area),
        "nc_major_axis_ratio": safe_div(nuc_major, cyt_major),
        "nc_minor_axis_ratio": safe_div(nuc_minor, cyt_minor),
        "nc_equivalent_diameter_ratio": safe_div(nuc_eq, cyt_eq),
        "cytoplasm_without_nucleus_fraction": safe_div(max(cyt_area - nuc_area, 0.0), cyt_area),
        "nucleus_minus_cytoplasm_eccentricity": record["nucleus_eccentricity"] - record["cytoplasm_eccentricity"],
        "nucleus_minus_cytoplasm_solidity": record["nucleus_solidity"] - record["cytoplasm_solidity"],
        "nucleus_minus_cytoplasm_extent": record["nucleus_extent"] - record["cytoplasm_extent"],
        "nucleus_cytoplasm_orientation_gap_deg": normalize_orientation_gap(
            record["nucleus_orientation_deg"], record["cytoplasm_orientation_deg"]
        ),
    }
    for channel in CHANNELS:
        derived[f"nc_{channel}_avg_intensity_ratio"] = safe_div(
            record[f"nucleus_{channel}_avg_intensity"],
            record[f"cytoplasm_{channel}_avg_intensity"],
        )
        derived[f"nucleus_minus_cytoplasm_{channel}_entropy"] = (
            record[f"nucleus_{channel}_entropy"] - record[f"cytoplasm_{channel}_entropy"]
        )
        derived[f"nucleus_minus_cytoplasm_{channel}_contrast"] = (
            record[f"nucleus_{channel}_avg_contrast"] - record[f"cytoplasm_{channel}_avg_contrast"]
        )
    return derived


def format_md_table(rows: Sequence[Tuple[str, str]]) -> str:
    lines = ["| 字段 | 含义 |", "| --- | --- |"]
    lines.extend([f"| `{name}` | {desc} |" for name, desc in rows])
    return "\n".join(lines)


def write_schema(schema_path: Path, total_rows: int, counts: Dict[str, int], derived_names: Sequence[str]) -> None:
    base_rows = [
        ("class_key", "类别短名，来自官方特征文件前缀。"),
        ("class_name", "类别显示名。"),
        ("image_id", "官方 cluster/single-cell image id。"),
        ("cell_id", "图内细胞实例 id。"),
    ]
    base_rows.extend((name, "官方 cytoplasm ROI 特征。") for name in CYTOPLASM_FEATURE_NAMES)
    base_rows.extend((name, "官方 nucleus ROI 特征。") for name in NUCLEUS_FEATURE_NAMES)
    derived_rows = [
        ("nc_area_ratio", "核面积 / 细胞质面积。"),
        ("cytoplasm_to_nucleus_area_ratio", "细胞质面积 / 核面积。"),
        ("nc_major_axis_ratio", "核长轴 / 细胞质长轴。"),
        ("nc_minor_axis_ratio", "核短轴 / 细胞质短轴。"),
        ("nc_equivalent_diameter_ratio", "核等效直径 / 细胞质等效直径。"),
        ("cytoplasm_without_nucleus_fraction", "去核后剩余细胞质面积占比。"),
        ("nucleus_minus_cytoplasm_eccentricity", "核偏心率减去细胞质偏心率。"),
        ("nucleus_minus_cytoplasm_solidity", "核 solidity 减去细胞质 solidity。"),
        ("nucleus_minus_cytoplasm_extent", "核 extent 减去细胞质 extent。"),
        ("nucleus_cytoplasm_orientation_gap_deg", "核与细胞质主轴方向差，归一化到 0-90 度。"),
        ("nc_r_avg_intensity_ratio", "R 通道核/胞平均强度比。"),
        ("nc_g_avg_intensity_ratio", "G 通道核/胞平均强度比。"),
        ("nc_b_avg_intensity_ratio", "B 通道核/胞平均强度比。"),
        ("nucleus_minus_cytoplasm_r_entropy", "R 通道核 entropy - 胞 entropy。"),
        ("nucleus_minus_cytoplasm_g_entropy", "G 通道核 entropy - 胞 entropy。"),
        ("nucleus_minus_cytoplasm_b_entropy", "B 通道核 entropy - 胞 entropy。"),
        ("nucleus_minus_cytoplasm_r_contrast", "R 通道核 contrast - 胞 contrast。"),
        ("nucleus_minus_cytoplasm_g_contrast", "G 通道核 contrast - 胞 contrast。"),
        ("nucleus_minus_cytoplasm_b_contrast", "B 通道核 contrast - 胞 contrast。"),
    ]
    missing = [name for name in derived_names if name not in {x[0] for x in derived_rows}]
    if missing:
        raise ValueError(f"Schema doc is missing derived fields: {missing}")

    lines = [
        "# SIPaKMeD Morphology Features",
        "",
        "该文件由 `scripts/build_sipakmed_morphology_features.py` 自动生成。",
        "",
        f"- 总样本数：`{total_rows}`",
        "- 数据来源：官方 `Features_CELL/*.dat`（细胞质/细胞核 ROI 特征）",
        "- 说明：官方 PDF 仅说明了三通道强度/纹理特征，但未显式写出通道顺序；这里按常见 `R/G/B` 顺序命名。",
        "",
        "## 类别样本数",
        "",
    ]
    for class_name, count in counts.items():
        lines.append(f"- `{class_name}`: `{count}`")

    lines.extend(
        [
            "",
            "## 字段说明",
            "",
            "### 基础字段与官方 ROI 特征",
            "",
            format_md_table(base_rows),
            "",
            "### 派生形态学指标",
            "",
            format_md_table(derived_rows),
            "",
        ]
    )
    schema_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    feature_root = dataset_root / "Features_CELL"
    if not feature_root.exists():
        raise FileNotFoundError(f"Features_CELL not found: {feature_root}")

    rows: List[Dict[str, float | int | str]] = []
    counts: Dict[str, int] = {}

    for class_key, class_name in CLASS_SPECS:
        cyt_path = feature_root / f"{class_key}_CYTOPLASM_FEAT.dat"
        nuc_path = feature_root / f"{class_key}_NUCLEI_FEAT.dat"
        cyt_records = parse_region_rows(cyt_path, CYTOPLASM_FEATURE_NAMES)
        nuc_records = parse_region_rows(nuc_path, NUCLEUS_FEATURE_NAMES)

        if set(cyt_records) != set(nuc_records):
            missing_in_nuc = sorted(set(cyt_records) - set(nuc_records))[:10]
            missing_in_cyt = sorted(set(nuc_records) - set(cyt_records))[:10]
            raise ValueError(
                f"{class_key}: cyt/nuc keys mismatch. Missing in nuclei: {missing_in_nuc}; missing in cytoplasm: {missing_in_cyt}"
            )

        counts[class_name] = len(cyt_records)
        for key in sorted(cyt_records):
            image_id, cell_id = key
            record: Dict[str, float | int | str] = {
                "class_key": class_key.lower(),
                "class_name": class_name,
                "image_id": image_id,
                "cell_id": cell_id,
            }
            record.update(cyt_records[key])
            record.update(nuc_records[key])
            record.update(derive_features(record))  # type: ignore[arg-type]
            rows.append(record)

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    schema_path = Path(args.schema_md).resolve()
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    derived_names = list(derive_features(rows[0]).keys())  # type: ignore[arg-type]
    write_schema(schema_path, len(rows), counts, derived_names)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved schema: {schema_path}")
    print(f"Total rows: {len(rows)}")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
