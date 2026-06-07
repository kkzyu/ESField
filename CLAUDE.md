# CLAUDE.md

ESField: Protein pocket site-aware energy guidance for 3D molecular generation (MVP).

## 新会话快速启动

1. **本文件** — 项目概览、路径、当前状态
2. `README/editable/任务看板.md` — 最新任务进度
3. `README/editable/实验运行记录.md` — 实验记录模板

## 当前状态 (2026-06-04)

- **GPU**: RTX 4090D 24GB 可用
- **代码**: ESField v7 全部完成, 76/76 测试通过 (9 原始 + 67 v7 新增)
- **实验状态**: ✅ 全部完成 (10 口袋生成 + 消融 + 多样性 + Vina 对接)
- **Base Generator**: DrugFlow (flow matching, 12.1M params)
- **核心成果**:
  - v7.1 在 7/10 口袋上实现 DirectOcc > 0% (v6-D.2 baseline = 0%)
  - 消融: λ=5.0 最佳, 硬覆盖 vs 软约束差异不显著
  - 多样性: v7.1 在所有口袋上 Vendi 高于 baseline (+27-45%)
  - 结合能: 占据组 Vina 得分不优于未占据组 (3mfw p=0.067 趋势为反向)
- **完整报告**: `docs/FINAL_REPORT_2026-06-04.md`
- **论文材料**: `paper/` (figures, tables, methods)
- **下一步**: 论文撰写

## 目录结构

```
/root/
├── ESField/                          # ESField 项目
│   ├── src/                          # 源代码
│   │   ├── guidance/                 # 推断期引导模块
│   │   │   ├── energy_guidance.py    # v4/v5 learned potential guidance
│   │   │   ├── flow_matching_guidance.py
│   │   │   ├── lambda_schedule.py
│   │   │   ├── latent_guidance.py    # ★ v7: SiteCompatibilityEnergy + Metadiffusion guidance
│   │   │   ├── kinetic_trajectory_shaping.py  # ★ v7: KTS scheduler
│   │   │   └── two_stage_generation.py  # ★ v7: Phase1 Occupy + Phase2 Connect
│   │   ├── evaluation/               # 评估指标
│   │   │   ├── posu.py               # POSU-v2.1 / HEWU / SWScore
│   │   │   ├── site_occupancy.py     # ★ v7: direct_occupancy_rate, best_compatible_distance
│   │   │   ├── site_matching.py
│   │   │   ├── molecule_quality.py
│   │   │   └── ...
│   │   ├── models/
│   │   │   ├── analytic_esfield.py   # v6-D.2 analytic field (coordinate-only, NO-GO)
│   │   │   ├── potential_network.py  # v4/v5 learned potential
│   │   │   └── ...
│   │   ├── site_detection/
│   │   ├── data/
│   │   ├── generation/
│   │   ├── training/
│   │   ├── utils/
│   │   └── visualization/
│   ├── scripts/                      # 实验脚本
│   │   ├── drugflow_esfield_guide.py # DrugFlow + ESField 主脚本
│   │   ├── run_v7_two_stage.py       # ★ v7: 两阶段生成 CLI
│   │   ├── run_v6d2_actionable_test.py
│   │   ├── targetdiff_esfield_guide.py
│   │   └── ...
│   ├── configs/
│   │   ├── v7_config.yaml            # ★ v7: 超参数配置
│   │   └── mvp.yaml
│   ├── tests/
│   │   ├── test_latent_guidance.py   # ★ v7: 39 tests
│   │   ├── test_two_stage.py         # ★ v7: 28 tests
│   │   └── ...
│   ├── experiments/                  # 实验结果
│   │   └── pdbbind_water_sites/
│   ├── docs/
│   │   ├── v7_实现与设计报告_2026-06-04.md  # ★ v7: 实现报告
│   │   └── v6-D_实现与诊断报告_2026-05-21.md
│   └── README/                       # 过程文档
```

## 关键命令

```bash
# 单元测试
cd /root/ESField && PYTHONPATH=src python -m unittest discover -s tests
# 76 passed (9 原始 + 67 v7), 10 pre-existing errors in test_analytic_esfield.py

# ============================================================
# v7 两阶段生成 (新)
# ============================================================

# v7 full pipeline
cd /root/ESField
PYTHONPATH=src python scripts/run_v7_two_stage.py \
  --protein-pdb <pdb> --ref-ligand <sdf> \
  --site-map <site_map.json> \
  --output-dir experiments/v7/<pocket_name> \
  --full-mol-size 25 --n-phase2-samples 5 \
  --lambda-early 0.5 --lambda-late 0.1

# v7 KTS sweep
for a0 in 0.01 0.05 0.1; do
  PYTHONPATH=src python scripts/run_v7_two_stage.py \
    --protein-pdb <pdb> --ref-ligand <sdf> \
    --site-map <site_map.json> \
    --output-dir experiments/v7/sweep/kts_a${a0} \
    --kts-alpha0 $a0
done

# v7 仅测试 Phase 1 (占位)
PYTHONPATH=src python -c "
from guidance.latent_guidance import *
from guidance.two_stage_generation import *
import json
site_map = json.load(open('<site_map.json>'))
energy = build_site_energy_from_map(site_map, sigma_distance=3.0)
print(f'Registered {energy.n_sites} HEW sites')
# ... 见 scripts/run_v7_two_stage.py 完整流程
"

# ============================================================
# DrugFlow baseline (单蛋白)
# ============================================================
cd /root/baselines/DrugFlow/code/DrugFlow-main
python src/generate.py --protein <pdb> --ref_ligand <sdf> \
  --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
  --n_samples 20 --output out.sdf

# DrugFlow + ESField learned potential (v5)
cd /root/ESField
python scripts/drugflow_esfield_guide.py \
  --protein-pdb <pdb> --ref-ligand <sdf> \
  --site-map <site_map.json> \
  --potential-ckpt experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt \
  --output-dir <dir> --esfield-lambda 2.0

# ============================================================
# Docking
# ============================================================
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs:$LD_LIBRARY_PATH"
vina --receptor rec.pdbqt --ligand lig.pdbqt --center_x X --center_y Y --center_z Z \
  --size_x 20 --size_y 20 --size_z 20 --out out.pdbqt
```
