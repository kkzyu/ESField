# ESField MVP 使用文档

更新时间：2026-05-14

本文档说明如何在租 GPU 前先完成无卡代码验证，以及租卡后如何按顺序运行 ESField MVP。

## 1. 当前代码能力

已实现模块：

- site map schema：`src/site_detection/site_schema.py`
- crystal water site builder：`src/site_detection/build_crystal_water_sites.py`
- fpocket parser：`src/site_detection/parse_fpocket.py`
- site merge/filter/top-K：`src/site_detection/merge_sites.py`
- atom-site pair schema/builder：`src/data/atom_site_schema.py`、`src/data/build_atom_site_pairs.py`
- MLP compatibility potential：`src/models/potential_network.py`
- loss、训练 dataset、训练 CLI：`src/models/losses.py`、`src/training/train_potential.py`
- inference-time coordinate guidance：`src/guidance/energy_guidance.py`
- RepMolFlow 命令级 baseline adapter：`src/generation/run_base_generator.py`
- site-specific metrics：`src/evaluation/site_matching.py`
- molecule quality proxy：`src/evaluation/molecule_quality.py`
- failure analysis：`src/evaluation/failure_analysis.py`
- PyMOL、3D scatter、HTML report：`src/visualization/`
- 安全索引 CrossDocked 压缩包：`scripts/index_crossdocked_archive.py`
- 无卡 toy pipeline：`scripts/run_toy_pipeline.py`

## 2. 无卡本地验证

在 `ESField` 目录下运行：

```bash
export PYTHONPATH="src"
python -m unittest discover -s tests
python -m compileall src scripts
python scripts/run_toy_pipeline.py --output-dir experiments/toy_pipeline --overwrite
```

预期：

- 单元测试通过；如果当前环境未安装 PyTorch，model/guidance 测试会被跳过。
- `experiments/toy_pipeline/` 下生成：
  - `toy_site_map.json`
  - `toy_atom_site_pairs.jsonl`
  - `toy_site_metrics.json`
  - `toy_site_metrics.csv`
  - `toy_failure_analysis.json`
  - `toy_sites.pml`
  - `toy_site_map.png` 或 fallback CSV
  - `toy_report.html`

这些 toy 结果只验证接口，不代表真实实验效果。

## 3. 安全查看 CrossDocked 压缩包

不要直接全量解压大压缩包。先索引少量样本：

```bash
python scripts/index_crossdocked_archive.py `
  --archive /root/autodl-tmp/data/downsampled_CrossDocked2020_v1.3.tgz `
  --output-csv experiments/archive_index/raw_index.csv `
  --limit 100 `
  --dry-run

python scripts/index_crossdocked_archive.py `
  --archive /root/autodl-tmp/data/downsampled_CrossDocked2020_v1.3_types.tgz `
  --output-csv experiments/archive_index/types_index.csv `
  --limit 20 `
  --dry-run
```

如果只想抽样解压前 100 个文件：

```bash
python scripts/index_crossdocked_archive.py `
  --archive /root/autodl-tmp/data/downsampled_CrossDocked2020_v1.3.tgz `
  --output-csv experiments/archive_index/raw_index.csv `
  --limit 100 `
  --extract `
  --extract-dir data\raw\crossdocked_sample
```

## 4. 构建 crystal water site map

对一个已解压的 receptor/ligand 复合物运行：

```bash
export PYTHONPATH="src"
python -m site_detection.build_crystal_water_sites `
  --protein-pdb data/raw/crossdocked_sample/CASE/example_rec.pdb `
  --ligand data/raw/crossdocked_sample/CASE/example_lig.sdf `
  --protein-id CASE `
  --ligand-id example_lig `
  --output data/site_maps/CASE_crystal_water.json `
  --pocket-radius 10 `
  --max-sites 20 `
  --overwrite
```

输出是 ESField site map JSON。水位点来自 crystal water + 规则近似；它不是 WATsite 结果。

## 5. 解析 fpocket 输出

先在服务器上运行 fpocket，得到类似 `*_out/` 的目录，再执行：

```bash
export PYTHONPATH="src"
python -m site_detection.parse_fpocket `
  --fpocket-dir data/fpocket/CASE_out `
  --protein-id CASE `
  --ligand-id example_lig `
  --pocket-center 0 0 0 `
  --output data/site_maps/CASE_fpocket.json `
  --max-sites 20 `
  --overwrite
```

`--pocket-center` 可以使用 ligand heavy atom centroid。后续可用脚本自动化，这里先保留显式参数，避免错误假设。

## 6. 合并 site map

```bash
export PYTHONPATH="src"
python -m site_detection.merge_sites `
  --inputs data/site_maps/CASE_crystal_water.json data/site_maps/CASE_fpocket.json `
  --output data/site_maps/CASE_merged.json `
  --merge-distance 1.0 `
  --max-sites 20 `
  --overwrite
```

## 7. 构建 atom-site pair

```bash
export PYTHONPATH="src"
python -m data.build_atom_site_pairs `
  --ligand data/raw/crossdocked_sample/CASE/example_lig.sdf `
  --site-map data/site_maps/CASE_merged.json `
  --output data/atom_site_pairs/CASE_pairs.jsonl `
  --split train `
  --negative-ratio 3 `
  --overwrite
```

输出 JSONL 中每行是一个 `AtomSitePair`，包含正样本、hard negatives、distance corruption 和 type corruption。

## 8. 训练 compatibility potential

需要 PyTorch。小规模 smoke run：

```bash
export PYTHONPATH="src"
python -m training.train_potential `
  --train-pairs data/atom_site_pairs/train_pairs.jsonl `
  --valid-pairs data/atom_site_pairs/valid_pairs.jsonl `
  --output-dir experiments/potential_smoke `
  --epochs 3 `
  --batch-size 128 `
  --device cpu
```

正式 GPU 运行时把 `--device cuda` 或不指定 device。输出：

- `run_metadata.json`
- `metrics.csv`
- `compatibility_potential_epoch_XXXX.pt`

核心验证指标：

- `valid_auc`
- `valid_ranking_accuracy`
- `valid_mean_pos_energy`
- `valid_mean_neg_energy`

目标是正样本平均 energy 低于负样本，AUC 明显高于 0.5。

## 9. Flow matching guidance 接入方式

通用 hook 位于：

```python
from guidance.flow_matching_guidance import apply_site_guidance_to_velocity
```

在 flow matching ODE step 中：

```python
guided_v, diag = apply_site_guidance_to_velocity(
    guidance,
    base_velocity,
    coordinates,
    t=t,
    site_map=site_map,
    atom_type_indices=atom_type_indices,
)
```

MVP 只引导坐标，不直接改 atom type logits。每个 sampling step 应 detach 当前坐标，避免跨 step 保留计算图。

注意：当前本地 `RepMolFlow-main.zip` 是 PropMolFlow/QM9 property-conditioned flow matching 示例，不是 pocket-conditioned SBDD 生成器。代码提供 baseline 命令适配器和通用 guidance hook；真正 guided generation 需要把 hook 插入目标生成器的采样循环。

## 10. 运行 baseline 命令适配器

如果 `/root/RepMolFlow` 下目前只有 `RepMolFlow-main.zip`，先在合适位置解压成 `RepMolFlow-main/`。当前已确认用户解压后的真实代码根目录是 `/root/RepMolFlow/RepMolFlow-main`；ESField 适配器也能接受外层 `/root/RepMolFlow` 并自动定位内层仓库。不要把解压后的外部仓库提交进 ESField。

只生成命令，不执行：

```bash
export PYTHONPATH="src"
python -m generation.run_base_generator `
  --repo-dir /root/RepMolFlow `
  --model-checkpoint /root/autodl-tmp/checkpoint/RepMolFlow-main/checkpoints/in-distribution/alpha/epoch=1845-step=721785.ckpt `
  --output-file experiments/baseline/alpha_in.sdf `
  --n-mols 100 `
  --property-name alpha `
  --properties-handle-method sum
```

确认环境和路径无误后再加 `--execute`。

## 11. 评估与结果表

site-specific metrics：

```bash
export PYTHONPATH="src"
python -m evaluation.site_matching `
  --ligand results/generated/CASE_sample.sdf `
  --site-map data/site_maps/CASE_merged.json `
  --output-json results/metrics/CASE_site_metrics.json `
  --output-csv results/metrics/CASE_site_metrics.csv
```

molecule quality proxy：

```bash
export PYTHONPATH="src"
python -m evaluation.molecule_quality `
  --inputs results/generated/CASE_sample.sdf `
  --output-json results/metrics/CASE_quality.json `
  --output-csv results/metrics/CASE_quality.csv
```

failure analysis：

```bash
export PYTHONPATH="src"
python -m evaluation.failure_analysis `
  --metrics results/metrics/CASE_site_metrics.csv `
  --output-json results/metrics/CASE_failure_analysis.json
```

## 12. 可视化与报告

PyMOL：

```bash
export PYTHONPATH="src"
python -m visualization.export_pymol `
  --site-map data/site_maps/CASE_merged.json `
  --receptor data/raw/crossdocked_sample/CASE/example_rec.pdb `
  --ligand results/generated/CASE_sample.sdf `
  --output results/visualization/CASE_sites.pml
```

3D scatter：

```bash
export PYTHONPATH="src"
python -m visualization.plot_site_map `
  --site-map data/site_maps/CASE_merged.json `
  --output results/visualization/CASE_site_map.png
```

HTML report：

```bash
export PYTHONPATH="src"
python -m visualization.report `
  --output results/reports/CASE_report.html `
  --title "ESField CASE Report" `
  --metric-files results/metrics/CASE_site_metrics.json results/metrics/CASE_failure_analysis.json `
  --notes "guided vs unguided case analysis"
```

## 13. 租卡运行注意

- 超过 10 分钟的训练或生成使用 `tmux`。
- 启动前把命令、配置、输出目录、随机种子写入 `README/editable/实验运行记录.md`。
- 先跑 1-3 个 pocket 的 smoke，再扩大到 100-500 个 pocket。
- 任何 full CrossDocked 遍历都必须带 `--limit` 或先做抽样索引。
- 不把 `data/`、`checkpoints/`、`results/`、`experiments/` 提交进 git。
