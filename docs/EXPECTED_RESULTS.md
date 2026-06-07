# ESField MVP 预期结果分析说明

更新时间：2026-05-14

本文档定义代码写完后，租卡运行 ESField MVP 时应该看到什么结果、如何制表、如何判断结果是否支持继续扩大实验。

## 1. 预期产物

每个 pocket 或 case 至少应产出：

- merged site map：`data/site_maps/{case_id}_merged.json`
- atom-site pairs：`data/atom_site_pairs/{case_id}_pairs.jsonl`
- potential checkpoint：`experiments/potential_*/compatibility_potential_epoch_XXXX.pt`
- generated molecules：`results/generated/{case_id}_*.sdf`
- site metrics：`results/metrics/{case_id}_site_metrics.csv/json`
- molecule quality：`results/metrics/{case_id}_quality.csv/json`
- failure analysis：`results/metrics/{case_id}_failure_analysis.json`
- visualization：
  - `{case_id}_sites.pml`
  - `{case_id}_site_map.png`
  - `{case_id}_report.html`

## 2. 势能网络预期

训练 `CompatibilityPotential` 后，优先看：

| 指标 | 预期方向 | MVP Go 标准 |
| --- | --- | --- |
| `valid_auc` | 越高越好 | >= 0.70 更理想；明显高于 0.5 才有意义 |
| `valid_ranking_accuracy` | 越高越好 | >= 0.65 更理想 |
| `valid_mean_pos_energy` | 越低越好 | 应低于负样本平均 energy |
| `valid_mean_neg_energy` | 越高越好 | 应高于正样本平均 energy |

如果 AUC 接近 0.5，说明 pair 标签、负样本或特征设计有问题。优先检查：

- 正样本是否只靠距离，缺少类型兼容约束。
- hard negative 是否太少。
- site type 分布是否严重不平衡。
- crystal water 规则是否产生大量低置信噪声位点。

## 3. Guidance 预期

guided generation 的诊断日志应至少记录：

- 每步或每若干步的 `E_total`
- `lambda_t`
- `grad_norm`
- 是否出现 NaN
- 生成分子的 validity/几何质量

预期现象：

- 中后期 guidance 开启后，`E_total` 应整体下降。
- `grad_norm` 不应长期贴着 clip 上限，否则说明势能过强或位点过密。
- guided 样本的 site metrics 应高于 unguided baseline。
- validity、PoseBusters pass、SA score、QED 不应明显恶化。

若生成分子塌缩到少数位点：

- 降低 `guidance.lambda_max`。
- 推迟 `guidance.guidance_start`。
- 降低 `site_detection.max_sites_per_pocket` 或提高 `min_confidence`。
- 后续加入 site capacity penalty。

## 4. Site-specific metrics 预期

核心指标：

| 指标 | 含义 | 预期方向 |
| --- | --- | --- |
| `site_occupancy_rate` | 有任意配体原子占据的目标位点比例 | guided > unguided |
| `correct_atom_site_matching_rate` | 被占据位点中原子类型兼容比例 | guided > unguided |
| `high_energy_water_replacement_rate` | 高能水位点被兼容原子替代比例 | guided > unguided |
| `stable_water_preservation_penalty` | 稳定水被不兼容原子占据比例 | guided < unguided |
| `hydrophobic_cavity_filling_score` | 疏水空腔被疏水原子填充的软分数 | guided > unguided |

MVP 最核心目标不是 docking score，而是至少一个 site-specific metric 稳定改善，并且化学/几何质量不明显变差。

建议主表：

| 方法 | SOR | CASMR | HEWRR | SWPP ↓ | HCFS | Validity | PoseBusters | QED | SA | Vina |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base flow matching | | | | | | | | | | |
| Random site guidance | | | | | | | | | | |
| Hand-crafted potential | | | | | | | | | | |
| ESField learned potential | | | | | | | | | | |

其中 `SWPP` 越低越好，其余 site metrics 通常越高越好。

## 5. 可视化预期

每个代表性 case 至少输出：

1. Site map 可视化：高能水、稳定水、疏水空腔用不同颜色显示。
2. Unguided vs guided ligand pose 对比。
3. 被改善的 atom-site match 标注。
4. 失败 case：例如 stable water 被占据、clash 增加或原子塌缩。

PyMOL 脚本颜色约定：

- `high_energy_water`：red
- `stable_water`：blue
- `hydrophobic_cavity`：yellow
- ligand：green
- receptor：gray

## 6. Go/No-Go 判断

Go：

- potential validation：AUC >= 0.70 或 ranking accuracy >= 0.65。
- guidance stability：无大规模 NaN、无明显原子塌缩。
- site metrics：SOR 或 CASMR 等核心指标有稳定提升。
- quality：validity / PoseBusters pass 下降不超过约 5 个百分点。
- docking proxy：Vina median 不明显变差。
- interpretability：至少 3 个 case 图能解释 guided 的局部变化。

No-Go：

- AUC 接近 0.5。
- guidance 后 validity 或 PoseBusters 明显下降。
- site metrics 没有稳定提升。
- docking 或几何质量明显变差且无法通过降低 guidance 修复。
- 结果主要来自生成更大、更疏水的分子，而不是更好的位点利用。

## 7. 报告写法建议

当前阶段不要写“降低真实结合自由能”。推荐表述：

> ESField 在 MVP 中验证的是 inference-time site-aware learned compatibility guidance 是否能提高物理机会位点利用率，同时保持生成分子的基础化学和几何质量。Vina、QED、SA 和 PoseBusters 是代理指标与质量门控，不等同于真实 binding free energy。

如果结果好：

- 强调 site-specific metrics 和可视化 case。
- 把 docking score 作为辅助指标。
- 报告 failure analysis，说明方法边界。

如果结果一般：

- 收缩为 proof-of-concept。
- 分析 site detection 噪声、负样本、guidance strength 和 base generator 接口限制。
- 下一轮优先增强 hard negative 和 pocket-conditioned generator 接入。

