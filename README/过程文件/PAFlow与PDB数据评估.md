# PAFlow Base Generator 与 PDB 数据集评估

日期：2026-05-16

## 一、PAFlow 评估

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 名称 | PAFlow |
| 论文 | NeurIPS 2025 "Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number" |
| 路径 | `/root/PAFlow-main` |
| 模型类型 | Pocket-conditioned 3D flow matching |
| 数据 | CrossDocked2020 v1.1, RMSD<1.0, pocket 10Å |
| 框架 | PyTorch + PyTorch Geometric |

### 1.2 与 ESField 的匹配度分析

**匹配点：**
- **Pocket-conditioned**：原生支持蛋白口袋条件生成，不是 QM9 模型
- **Flow matching**：与 ESField 最初设计目标一致（ODE 积分、velocity field）
- **CrossDocked2020**：使用与 ESField 相同的数据集
- **已有 guidance 机制**：内置 binding affinity guidance（`pos_grad_w`、`v_grad_w`），证明采样循环支持梯度引导
- **单口袋生成**：`sample_for_pocket.py` 支持对任意给定口袋生成分子
- **完整的评估体系**：Vina docking、PoseBusters、分子质量指标

**不匹配点（需要适配）：**
- PAFlow 的 guidance 是 affinity-based，需要替换/结合为 ESField 的 site-based guidance
- PAFlow 的原子数预测需要口袋体积/面积（pyKVFinder），ESField 不需要这个
- PAFlow 已有自己的原子类型词汇表，ESField 需要映射到 PAFlow 的类型系统

### 1.3 关键技术架构

**模型结构：`ScorePosNet3D_guided_flow`**（`models/molopt_score_model_guide.py`）

```
输入：protein_pos, protein_v, batch_protein, ligand_xt, ligand_vt, time_step
  ↓
Protein Encoder (UniTransformer/EGNN)
  ↓
Ligand Encoder (UniTransformer/EGNN)  
  ↓ 交叉注意力
Joint Representation
  ↓
pred_ligand_pos ← 预测清洁坐标 x1
pred_ligand_v   ← 预测清洁原子类型 v1
final_affinity_pred ← per-atom affinity → scatter_mean → molecule affinity
```

**采样循环：`sample_guided_flow_VP()`**（50 steps，VP path）

```python
for i in reversed(range(50)):
    with torch.enable_grad():
        ligand_xt = ligand_xt.detach().requires_grad_(True)
        ligand_vt = ligand_vt.detach().requires_grad_(True)
        
        preds = model(protein, ligand_xt, ligand_vt, t)
        pred_affinity = preds['final_affinity_pred']
        
        # === 现有 guidance（affinity-based） ===
        log_p_y_posterior = log P(high_affinity | current_state)
        pos_guidance = grad(log_p_y_posterior, ligand_xt)
        v_guidance = grad(log_p_y_posterior, ligand_vt)
        
        # === ODE step ===
        dx = VP_field(x1_pred, xt, t) + para_x * pos_guidance * pos_grad_w
        dv = (v1_pred - vt) * alpha + para_v * v_guidance * v_grad_w
        
        ligand_xt = ligand_xt + dx * dt
        ligand_vt = ligand_vt + dv * dt
```

### 1.4 ESField 注入点

**注入位置**：在 `sample_guided_flow_VP()` 的第 701-703 行之后。当前：
```python
# Line 701-703
log_p_y_posterior = self.log_p_y_posterior(pred_affinity, 1)
pos_guidance = torch.autograd.grad(log_p_y_posterior, ligand_xt, ...)[0]
v_guidance = torch.autograd.grad(log_p_y_posterior, ligand_vt, ...)[0]
```

**注入 ESField guidance**：
```python
# 新增：ESField site-based guidance
esfield_energy = esfield_guidance.total_energy(
    ligand_xt, site_map=site_map,
    atom_type_probs=F.softmax(preds['pred_ligand_v'], dim=-1)
)
esfield_pos_grad = torch.autograd.grad(esfield_energy, ligand_xt, ...)[0]
esfield_pos_grad = clip_by_norm(esfield_pos_grad, max_norm=1.0)

lambda_t = esfield_lambda_schedule(t)
pos_guidance = pos_guidance + esfield_pos_grad * lambda_t
```

**关键风险**：
- PAFlow 在每步 `detach().requires_grad_(True)` 后计算梯度，与 ESField 的 `EnergyGuidance.coordinate_gradient()` 内部 detach 逻辑兼容
- 需要注意两种 guidance 的尺度匹配（pos_grad_w=350 来自论文，esfield lambda_max≈0.1）
- 需要对 PAFlow 的原子类型索引与 ESField 的 `ATOM_TYPE_VOCAB` 做映射

### 1.5 待获取的资源

- [ ] PAFlow 预训练 checkpoint：`./pretrained_models/pretrained_flow.pt`（需从 Google Drive 下载）
- [ ] PAFlow 原子数预测模型：`./pretrained_models/atom_num_model.pt`
- [ ] CrossDocked pocket10 数据集：`./data/crossdocked_v1.1_rmsd1.0_pocket10`
- [ ] 数据 split 文件：`./data/crossdocked_pocket10_pose_split.pt`
- [ ] pyKVFinder 环境（用于计算口袋体积/面积）

## 二、PDB 数据集评估

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 路径 | `/root/autodl-tmp/data/PDB/P-L.tar.gz` |
| 大小 | 3.2 GB |
| 内容 | PDBbind protein-ligand 结构数据 |
| 目录结构 | `P-L/{year_range}/{pdb_code}/` |
| 每案例文件 | `protein.pdb`, `pocket.pdb`, `ligand.sdf`, `ligand.mol2` |

### 2.2 Crystal Water 可用性

- **protein.pdb**：完整的蛋白结构文件，预期包含 crystal water（HOH/WAT 残基）
- **pocket.pdb**：口袋周围 10Å 提取的结构，可能不包含水分子

**结论**：`protein.pdb` 文件中的 crystal water 可以用于 ESField 的 `build_crystal_water_sites.py`，解决 CrossDocked 受体不含水分子的问题。

### 2.3 与 CrossDocked 的关系

PDBbind 是 CrossDocked 的上游数据源之一（CrossDocked 通过交叉对接扩展了 PDBbind）。两者的 PDB 结构可以直接对应。ESField 可以：
1. 用 PDB 的 `protein.pdb`（含 crystal water）构建 water site map
2. 用 CrossDocked 的受体（口袋）做生成器条件
3. 将 site map 坐标对齐到 CrossDocked 口袋坐标系

### 2.4 数据规模

PDBbind general set (PL) 通常包含 ~2000-3000 个 protein-ligand 复合物，足够 MVP 使用。

## 三、总体评估结论

**PAFlow 是 RepMolFlow 的优秀替代方案：**
- 原生 pocket-conditioned，无需从零训练 SBDD 生成器
- Flow matching 架构与 ESField 的设计理念一致
- 已有 guidance 机制，注入点明确
- NeurIPS 2025 论文，方法质量有保证

**PDB 数据集解决了 crystal water 缺失问题：**
- protein.pdb 包含完整蛋白结构和水分子
- 可与 CrossDocked 口袋配合使用

**下一步关键工作：**
1. 获取 PAFlow 预训练模型（Google Drive 下载）
2. 获取 PAFlow 的 CrossDocked pocket10 数据集
3. 建立 ESField atom_type ↔ PAFlow atom_type 的映射
4. 修改 `sample_guided_flow_VP()` 注入 ESField site guidance
5. 从 P-L.tar.gz 提取选定案例的 crystal water
6. 端到端测试：PDB water site → PAFlow pocket-conditioned generation → ESField guided sampling
