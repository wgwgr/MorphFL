# SIPaKMeD Morphology Features

该文件由 `scripts/build_sipakmed_morphology_features.py` 自动生成。

- 总样本数：`4049`
- 数据来源：官方 `Features_CELL/*.dat`（细胞质/细胞核 ROI 特征）
- 说明：官方 PDF 仅说明了三通道强度/纹理特征，但未显式写出通道顺序；这里按常见 `R/G/B` 顺序命名。

## 类别样本数

- `Dyskeratotic`: `813`
- `Koilocytotic`: `825`
- `Metaplastic`: `793`
- `Parabasal`: `787`
- `Superficial-Intermediate`: `831`

## 字段说明

### 基础字段与官方 ROI 特征

| 字段 | 含义 |
| --- | --- |
| `class_key` | 类别短名，来自官方特征文件前缀。 |
| `class_name` | 类别显示名。 |
| `image_id` | 官方 cluster/single-cell image id。 |
| `cell_id` | 图内细胞实例 id。 |
| `cytoplasm_area` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_major_axis_length` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_minor_axis_length` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_eccentricity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_orientation_deg` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_equivalent_diameter` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_solidity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_extent` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_avg_intensity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_avg_contrast` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_smoothness` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_uniformity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_third_moment` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_r_entropy` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_avg_intensity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_avg_contrast` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_smoothness` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_uniformity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_third_moment` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_g_entropy` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_avg_intensity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_avg_contrast` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_smoothness` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_uniformity` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_third_moment` | 官方 cytoplasm ROI 特征。 |
| `cytoplasm_b_entropy` | 官方 cytoplasm ROI 特征。 |
| `nucleus_area` | 官方 nucleus ROI 特征。 |
| `nucleus_major_axis_length` | 官方 nucleus ROI 特征。 |
| `nucleus_minor_axis_length` | 官方 nucleus ROI 特征。 |
| `nucleus_eccentricity` | 官方 nucleus ROI 特征。 |
| `nucleus_orientation_deg` | 官方 nucleus ROI 特征。 |
| `nucleus_equivalent_diameter` | 官方 nucleus ROI 特征。 |
| `nucleus_solidity` | 官方 nucleus ROI 特征。 |
| `nucleus_extent` | 官方 nucleus ROI 特征。 |
| `nucleus_r_avg_intensity` | 官方 nucleus ROI 特征。 |
| `nucleus_r_avg_contrast` | 官方 nucleus ROI 特征。 |
| `nucleus_r_smoothness` | 官方 nucleus ROI 特征。 |
| `nucleus_r_uniformity` | 官方 nucleus ROI 特征。 |
| `nucleus_r_third_moment` | 官方 nucleus ROI 特征。 |
| `nucleus_r_entropy` | 官方 nucleus ROI 特征。 |
| `nucleus_g_avg_intensity` | 官方 nucleus ROI 特征。 |
| `nucleus_g_avg_contrast` | 官方 nucleus ROI 特征。 |
| `nucleus_g_smoothness` | 官方 nucleus ROI 特征。 |
| `nucleus_g_uniformity` | 官方 nucleus ROI 特征。 |
| `nucleus_g_third_moment` | 官方 nucleus ROI 特征。 |
| `nucleus_g_entropy` | 官方 nucleus ROI 特征。 |
| `nucleus_b_avg_intensity` | 官方 nucleus ROI 特征。 |
| `nucleus_b_avg_contrast` | 官方 nucleus ROI 特征。 |
| `nucleus_b_smoothness` | 官方 nucleus ROI 特征。 |
| `nucleus_b_uniformity` | 官方 nucleus ROI 特征。 |
| `nucleus_b_third_moment` | 官方 nucleus ROI 特征。 |
| `nucleus_b_entropy` | 官方 nucleus ROI 特征。 |

### 派生形态学指标

| 字段 | 含义 |
| --- | --- |
| `nc_area_ratio` | 核面积 / 细胞质面积。 |
| `cytoplasm_to_nucleus_area_ratio` | 细胞质面积 / 核面积。 |
| `nc_major_axis_ratio` | 核长轴 / 细胞质长轴。 |
| `nc_minor_axis_ratio` | 核短轴 / 细胞质短轴。 |
| `nc_equivalent_diameter_ratio` | 核等效直径 / 细胞质等效直径。 |
| `cytoplasm_without_nucleus_fraction` | 去核后剩余细胞质面积占比。 |
| `nucleus_minus_cytoplasm_eccentricity` | 核偏心率减去细胞质偏心率。 |
| `nucleus_minus_cytoplasm_solidity` | 核 solidity 减去细胞质 solidity。 |
| `nucleus_minus_cytoplasm_extent` | 核 extent 减去细胞质 extent。 |
| `nucleus_cytoplasm_orientation_gap_deg` | 核与细胞质主轴方向差，归一化到 0-90 度。 |
| `nc_r_avg_intensity_ratio` | R 通道核/胞平均强度比。 |
| `nc_g_avg_intensity_ratio` | G 通道核/胞平均强度比。 |
| `nc_b_avg_intensity_ratio` | B 通道核/胞平均强度比。 |
| `nucleus_minus_cytoplasm_r_entropy` | R 通道核 entropy - 胞 entropy。 |
| `nucleus_minus_cytoplasm_g_entropy` | G 通道核 entropy - 胞 entropy。 |
| `nucleus_minus_cytoplasm_b_entropy` | B 通道核 entropy - 胞 entropy。 |
| `nucleus_minus_cytoplasm_r_contrast` | R 通道核 contrast - 胞 contrast。 |
| `nucleus_minus_cytoplasm_g_contrast` | G 通道核 contrast - 胞 contrast。 |
| `nucleus_minus_cytoplasm_b_contrast` | B 通道核 contrast - 胞 contrast。 |
