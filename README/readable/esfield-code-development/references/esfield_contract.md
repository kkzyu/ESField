# ESField 实现契约

实现项目代码时阅读本文件。它记录跨模块必须一致的约定；如果项目级决策变化，要同步更新。

## MVP 边界

- 目标：验证 `offline site map + learned atom-site potential + inference-time coordinate guidance` 能否提升 site-specific metrics，同时不破坏生成质量。
- 数据：从 downsampled CrossDocked 和/或小规模 PDBbind 子集起步。
- 生成器：第一版面向 flow matching generator，但在实际仓库接入前，wrapper 不得假设某个固定 API。
- 位点类型：`high_energy_water`、`stable_water`、`hydrophobic_cavity`。
- Guidance：MVP 只引导坐标。
- 生成目标中不加入合成可及性约束，但评估时报告 SA score。

## 语言契约

- 所有说明性文档、阶段总结、运行记录、任务看板、handoff 和失败分析默认用中文。
- 代码标识符、配置键、CLI 参数、schema 字段名使用英文。
- 日志可以中英混合：字段名英文，解释文本中文。
- 论文和研究计划相关文本用中文，除非用户明确要求英文投稿稿。

## 必要 Schema

Site map record 必须保留：

- `schema_version`
- `protein_id`
- `ligand_id`
- `pocket_center`
- `coordinate_frame`
- `sites`

每个 site 必须保留：

- `site_id`
- `site_type`
- `center`
- `radius`
- `score`
- `confidence`
- `source`
- `features`

Atom-site pair record 必须保留：

- `schema_version`
- `protein_id`
- `ligand_id`
- `atom_index`
- `atom_type`
- `atomic_number`
- `site_id`
- `site_type`
- `relative_position`
- `distance`
- `site_radius`
- `label`
- `label_strength`
- `negative_type`
- `split`

## 路径契约

使用配置键或环境变量，不写死路径。

推荐配置键：

```yaml
paths:
  data_root: ${ESFIELD_DATA_DIR:-data}
  run_root: ${ESFIELD_RUN_DIR:-experiments}
  checkpoint_root: ${ESFIELD_CHECKPOINT_DIR:-checkpoints}
```

大型输出放在被 `.gitignore` 忽略的目录：

- `data/`
- `checkpoints/`
- `results/`
- `experiments/`
- `logs/`

## 脚本契约

任何会触碰大量文件的脚本都必须支持：

- `--config`
- `--limit`
- `--dry-run`
- `--overwrite`，默认 false
- 清晰进度日志
- 尽可能支持断点续跑

任何训练或生成运行必须保存：

- 配置快照
- 命令行
- git commit，如果可用
- 随机种子
- 环境摘要
- stdout/stderr 日志
- 中文运行摘要

## Smoke Test 契约

最低检查要求：

- Schema：toy JSON round trip。
- Site detection：一个 toy PDB-like example 或小 fixture。
- Atom-site pairs：正样本和 hard negative 数量确定可断言。
- Potential model：forward loss 有限，坐标梯度有限且非零。
- Guidance：输出 shape 等于输入 shape，梯度范数被裁剪。
- Metrics：toy molecule/site 样例返回预期的 SOR/CASMR 类指标。

## 命名契约

稳定使用这些英文类名：

- `Site`
- `SiteMap`
- `AtomSitePair`
- `CompatibilityPotential`
- `EnergyGuidance`
- `SiteMatchingMetrics`

避免误导性命名：

- 不要把 compatibility potential 命名为 `free_energy`。
- 不要把 proxy docking metrics 命名为 `binding_affinity`。
- 不要把 crystal-water heuristics 命名为 `WATsite`，除非该结果确实由 WATsite 生成。
