# v7.1a — Flexible Anchor Annealing 实验指南

## 背景

v7.1 的硬覆盖（hard fix）虽然保证了锚原子在 Phase 2 全程固定在 HEW 位点，但最终结合能分析显示：占据 HEW 位点的分子 **并不优于** 未占据分子，尤其在 3mfw 口袋上呈显著更差的趋势（p=0.067, Cliff's δ=+0.47）。

**假说**：硬覆盖强制锚原子固定在非全局最优位置，导致整体分子构象次优。如果能在 Phase 2 前期保持约束（确保锚原子不漂移），后期逐步释放（允许局部微调），则可能改善结合能。

v7.1a 实现了这个"柔性锚点 annealing"策略。

## 算法

```
Phase 2 (总步数 T):
  ├── 步骤 0 .. N_fix-1 (前 70%):  硬覆盖 — 坐标强制设回 Phase 1 锚点值
  └── 步骤 N_fix .. T-1 (后 30%):  谐波约束 — E = k(t) * ||x - x0||^2
                                   其中 k(t) 从 10.0 线性退火到 0.0
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `fix_fraction` | 0.7 | 硬覆盖步数比例 |
| `restraint_start` | 10.0 | 谐波约束初始力常数 |
| `restraint_end` | 0.0 | 谐波约束最终力常数 |
| `ramp` | "linear" | 退火方式（linear / exponential） |

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/guidance/annealing_fix.py` | AnnealingAnchorFix 回调类 + 工厂函数 |
| `configs/v7_annealing.yaml` | v7.1a 实验配置文件 |
| `scripts/run_annealing_experiment.py` | 3mfw annealing 实验运行脚本 |
| `paper/manuscript.tex` | 论文初稿（bioRxiv 格式） |
| `paper/figures/generate_figures.py` | 论文图表生成脚本 |

## 快速开始

### 1. 环境准备

```bash
cd /root/ESField

# 确认 DrugFlow 已安装
ls /root/baselines/DrugFlow/code/DrugFlow-main/

# 确认 checkpoint
ls /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt

# 确认 site maps
ls experiments/pdbbind_water_sites/site_maps/3mfw_sites.json
```

### 2. 代码测试（无需 GPU）

```bash
# 测试 annealing 回调逻辑
cd /root/ESField
PYTHONPATH=src python -c "
from guidance.annealing_fix import AnnealingAnchorFix, create_anchor_callback
import torch

# 测试硬覆盖阶段
cb = AnnealingAnchorFix(
    anchor_indices=[0, 1],
    anchor_coords=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    total_steps=100,
    fix_fraction=0.7,
    verbose=True,
)
ligand = {'x': torch.randn(10, 3)}
ligand = cb(ligand, 0, 0.0)
print(f'Step 0 (hard fix): anchor[0] = {ligand[\"x\"][0]}')

# 测试退火阶段
for s in range(1, 100):
    ligand['x'] = torch.randn(10, 3)
    ligand = cb(ligand, s, 0.0)
print(f'Final k = {cb.current_k:.3f}')
print('All tests passed!')
"

# 运行 annealing 单元测试
cd /root/ESField
PYTHONPATH=src python src/guidance/annealing_fix.py
```

### 3. Dry Run（检查参数，无需 GPU）

```bash
cd /root/ESField
PYTHONPATH=src python scripts/run_annealing_experiment.py \
    --pocket 3mfw --n-mols 25 --conditions hard annealing
```

### 4. 完整实验运行（需要 GPU）

```bash
cd /root/ESField

# 确保 DrugFlow 已 patch（支持 post_step_callback）
PYTHONPATH=src python -c "from guidance.hard_fix import patch_drugflow_hardfix; patch_drugflow_hardfix()"

# 运行 annealing 实验
PYTHONPATH=src python scripts/run_annealing_experiment.py \
    --pocket 3mfw \
    --run \
    --n-mols 25 \
    --conditions hard annealing \
    --protein-pdb experiments/pdbbind_water_sites/3mfw/3mfw_protein.pdb \
    --ref-ligand experiments/pdbbind_water_sites/3mfw/3mfw_ligand.sdf \
    --site-map-json experiments/pdbbind_water_sites/site_maps/3mfw_sites.json \
    --output-dir experiments/annealing_3mfw
```

### 5. 运行 Vina 对接（实验后）

```bash
# 对每个条件生成的分子进行 Vina 对接
for condition in hard annealing; do
    python scripts/compute_vina_docking.py \
        --sdf experiments/annealing_3mfw/${condition}/3mfw_mols.sdf \
        --protein experiments/pdbbind_water_sites/3mfw/3mfw_protein.pdb \
        --output experiments/annealing_3mfw/${condition}/vina_scores.csv \
        --exhaustiveness 8
done

# 统计分析
python scripts/analyze_energy_trend.py \
    --results experiments/annealing_3mfw \
    --output experiments/annealing_3mfw/energy_analysis.json
```

## 生成论文图表

### 全部论文图表

```bash
cd /root/ESField
python paper/figures/generate_figures.py
```

生成以下文件：
- `paper/figures/fig1_pipeline_schematic.{pdf,png}` — 两阶段流程示意图
- `paper/figures/fig2_direct_occ_barplot.{pdf,png}` — DirectOcc 柱状图
- `paper/figures/fig3_ablation_heatmap.{pdf,png}` — 消融实验热图
- `paper/figures/fig4_vendi_diversity.{pdf,png}` — 多样性对比
- `paper/figures/fig5_vina_scatter.{pdf,png}` — 结合能散点图
- `paper/figures/figS1_v6_v7_comparison.{pdf,png}` — v6-D.2 vs v7.1 对比

### Annealing 对比图（实验后）

Annealing 对比图在 `run_annealing_experiment.py` 运行后自动生成：
- `experiments/annealing_3mfw/fig_annealing_comparison.{pdf,png}` — hard vs annealing vs soft 对比

### 编译论文

```bash
cd /root/ESField/paper
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## 预期结果

基于 v7.1 报告数据和算法分析，预期：

| 指标 | v7.1 hard | v7.1a annealing | 预期变化 |
|------|:---:|:---:|:---:|
| DirectOcc | ~12% | 8-12% | 略降（锚点可微调，可能偏离位点） |
| Mean Vina | -6.4 | -6.6 ~ -6.8 | 略改善（构象更优） |
| QED | 0.33 | 0.33-0.38 | 略改善（应变减少） |

如果 annealing 能在保持 DirectOcc > 5% 的同时将 Vina 改善 ≥ 0.5 kcal/mol，则证明柔性锚点策略的有效性，可作为 v7.2 的基础。

## 配置调优

### 调整退火速度

```yaml
# configs/v7_annealing.yaml
phase2:
  annealing:
    fix_fraction: 0.5      # 更早释放（50% 而非 70%）
    restraint_start: 20.0   # 更强的初始约束
    restraint_end: 0.0
    ramp: "exponential"     # 指数衰减（前期衰减快，后期慢）
```

### 仅针对特定口袋

```bash
# 在其他口袋上测试 annealing
PYTHONPATH=src python scripts/run_annealing_experiment.py \
    --pocket 2gni --run --n-mols 25 --conditions hard annealing \
    --site-map-json experiments/pdbbind_water_sites/site_maps/2gni_sites.json
```

## 集成到现有流水线

在现有 v7.1 代码中使用 annealing：

```python
from guidance.two_stage_generation import (
    TwoStageConfig, TwoStageGenerator,
    Phase1Config, Phase2Config,
)

# 使用 annealing 模式
p2cfg = Phase2Config(
    anchor_fix_mode="annealing",
    annealing_fix_fraction=0.7,
    annealing_restraint_start=10.0,
    annealing_restraint_end=0.0,
    lambda_late=0.1,
)

cfg = TwoStageConfig(phase1=Phase1Config(), phase2=p2cfg)
gen = TwoStageGenerator(cfg, model, site_map)
gen.to("cuda:0")

# Phase 1: 占位
anchors = gen.phase1_occupy(protein_data)

# Phase 2: 连接（自动使用 annealing 回调）
mols = gen.phase2_connect(protein_data, anchors, full_size)
```
