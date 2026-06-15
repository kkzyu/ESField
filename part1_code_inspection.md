# 第一部分：代码实现检查报告

## 1. 第一阶段（占据）实现细节

### 1.1 梯度类型：逐原子全梯度 ✓

第一阶段使用**逐原子全梯度**（每个原子独立计算 ∇_x E_site），而不是质心投影梯度。

**证据：**
- `_Phase1GuideFn.__call__()`（two_stage_generation.py:1275-1304）计算 `E_site = -Σ_i Σ_j compat_ij * exp(-d_ij²/(2σ²))`，其中求和独立地遍历所有原子-位点对。
- DrugFlow 内部通过自动微分计算 `∇_x guide_log_prob`。由于每个原子对求和的贡献是独立的，每个原子 `i` 得到：
  ```
  ∂E_site/∂x_i = Σ_j compat_ij * (x_i - c_j)/σ² * exp(-d_ij²/(2σ²))
  ```
- 这是**逐原子全梯度**——每个原子获得自己的梯度向量。

**结论：** 代码已经正确——使用逐原子全梯度，无需修改。

---

### 1.2 引导强度 λ_Phase1 和 KTS

**当前状态（与计划存在差异）：**
- `lambda_early` 默认值：**0.5**（而非计划中指定的 5.0）
  - 文件：`two_stage_generation.py` 第 84 行：`lambda_early: float = 0.5`
  - 同样在 `run_v7_two_stage.py` 第 107 行：`default=0.5`
- KTS 早期增强：**已启用**，参数如下：
  - `kts_alpha0 = 0.01`（早期增强）
  - `kts_beta0 = 0.01`（后期衰减）
  - `tau_split = 0.6`（转换点）
  - `k = 3.0`（指数刚度）
  - 文件：`kinetic_trajectory_shaping.py` 第 90-99 行

**计划要求：**
- λ_Phase1 应为 **5.0**（比当前默认值强 10 倍）
- 需要修改 → **lambda_early 应改为 5.0**

---

### 1.3 第一阶段原子数

- 固定为 **4 个原子**（`n_init_atoms: int = 4`，two_stage_generation.py 第 67 行）
- 每次运行不随机初始化——始终为 4

**结论：** 符合计划，4 个原子是正确的。

---

### 1.4 锚点选择标准

第一阶段之后，原子符合"锚点"资格的条件：
- **距离阈值：** `success_distance = 2.5` Å（第 72 行）
- **兼容性阈值：** `min_compatibility = 0.3`（第 77 行）
- 两个条件必须同时满足：`d ≤ 2.5 Å AND compat ≥ 0.3`

**选择策略**（第 104-110 行）：
- `"best_per_site"`：每个占据位点兼容性最好的原子
- `"all_compatible"`：所有符合标准的原子
- `"nearest_compatible"`：单个最近的兼容原子

**结论：** 符合计划，2.5 Å 和 0.3 的值正确。

---

### 1.5 锚点信息传递到第二阶段

通过 `AnchorAtoms` 数据类传递（two_stage_generation.py:221-246）：
```python
@dataclass
class AnchorAtoms:
    positions: torch.Tensor      # [n_anchors, 3] 坐标
    type_indices: torch.Tensor   # [n_anchors] 原子类型索引
    type_probs: torch.Tensor     # [n_anchors, n_atom_types] 软类型概率
    occupied_sites: list[int]    # 占据的 HEW 位点 ID
    compat_scores: list[float]   # 每个锚点的兼容性分数
    distances: list[float]       # 每个锚点到最近位点的距离
```

直接从 `phase1_occupy()` → 返回值 → `phase2_connect(anchors=...)` 传递。

**结论：** 完整。所有需要的信息（坐标、类型、HEW 位点 ID）都被保留。

---

## 2. 第二阶段（连接）实现细节

### 2.1 质心投影实现

**"kinematic" 模式**（`anchor_fix_mode == "kinematic"`）：
- 是的，质心投影已在 `KinematicAnchorGuidance._compute_com_site_gradient()` 中实现（kinematic_anchor.py:288-352）
- 计算锚点质心处的梯度，并将**纯平移**应用于所有锚点原子
- 第 247 行：`ligand["x"][idx_tensor] = anchor_x + correction.unsqueeze(0)` —— 相同的修正添加到每个锚点

**"hard" 模式**（默认）：
- 使用 `HardFixCallback` —— 每步直接覆盖锚点坐标
- 不计算质心梯度 —— 仅设置 `x_i = x_i_fixed`

**其他模式：**
- "annealing"：早期步骤硬固定，后期谐波衰减
- "soft"：仅谐波约束（通过 `TwoStageGuideFn`）

---

### 2.2 时间调度 λ(t)

在 `KinematicScheduler` 中（kinematic_anchor.py:42-94）：
- 公式（10）已实现：`λ(t) = λ_max * (1 - t)^2` 用于 "quadratic" 曲线
- **但 λ_max 默认值为 0.5**（而非计划指定的 1.0）
  - 第 56 行：类定义中 `lambda_max: float = 1.0`，但是……
  - 第 140 行：`KinematicAnchorGuidance.__init__` 中实际默认值 `lambda_max: float = 0.5`
  - Phase2Config 第 152 行：`kinematic_lambda_max: float = 0.5`

**计划要求：λ_max 应为 1.0，需要修改。**

---

### 2.3 硬固定基线实现

**当前实现："每步重置"**

`HardFixCallback.__call__()`（hard_fix.py:76-128）：
- 每个 ODE 步骤之后，锚点坐标被覆盖**回初始固定值**
- ODE 步骤**确实会先更新**锚点坐标，然后回调函数重置它们
- 第 112 行：`ligand["x"][idx_tensor] = self.anchor_coords[:len(valid_indices)]`

**计划要求：改为"完全锁定"**
- 锚点原子应**完全排除**在 ODE 更新之外
- 它们的坐标在整个第二阶段应保持不变
- 这需要修改 DrugFlow 的 simulate()，在 ODE 积分期间跳过锚点原子

**为何重要：**
- "每步重置"允许 ODE 更新锚点，然后强制它们返回 —— 这会制造人为的 KPE 尖峰
- "完全锁定"防止对锚点进行任何 ODE 更新 —— 更干净的基线，更公平的比较
- 当前实现可能**低估**硬固定的应变

---

## 3. E_site 实现

### 3.1 参数

- **σ（sigma_distance）：** **3.0** ✓（多个文件确认：latent_guidance.py:356，two_stage_generation.py:101）
- **τ：** 计划提到 τ=10，但代码库使用 KTS 的 `tau_split=0.6`（时间分数）。这些是不同的参数。计划中的 τ 可能指时间常数（未显式使用），而代码使用 σ=3.0 和 τ_split=0.6。
- **共享 σ：** 是的，所有 HEW 位点共享相同的 σ=3.0

**标记：** 计划询问 τ=10，但该参数在当前代码中未出现。需要澄清。

---

### 3.2 兼容性矩阵 M

**⚠️ 关键：存在两个冲突的版本！**

**版本 A** —— `latent_guidance.py` 第 101-154 行（由 `SiteCompatibilityEnergy` 使用）：
```python
# 4×11 numpy 数组等效：
COMPAT_MATRIX_A = np.array([
    # unknown  C_sp3  C_arom  N_don  N_acc  O_acc  S     P     halog  chrg  B
    [-0.5,     1.0,   1.0,   -0.5,  -0.5,  -0.5,  0.3,  -0.3,  1.0,  -1.0, -0.3],  # hydrophobic
    [-0.5,    -0.5,  -0.5,    1.0,   1.0,   1.0,  0.3,  -0.3, -0.5,  -1.0, -0.3],  # polar_unsat
    [ 0.0,     0.5,   0.5,    0.5,   0.5,   0.5,  0.5,   0.1,  0.5,  -0.5,  0.1],  # mixed
    [-0.5,     0.3,  -0.5,   -1.0,  -1.0,  -1.0, -0.3,  -0.5,  0.5,  -1.0, -0.3],  # buried
])
```

**版本 B** —— `analytic_esfield.py` 第 57-82 行（由 `AnalyticESFieldGuideV2` 使用）：
```python
COMPAT_MATRIX_B = np.array([
    # unknown  C_sp3  C_arom  N_don  N_acc  O_acc  S     P     halog  chrg  B
    [ 0.0,     1.0,   1.0,   -0.5,  -0.5,  -0.5,  0.3,  -0.3,  0.3,  -1.0,  0.0],  # hydrophobic
    [ 0.0,    -0.3,  -0.3,    1.0,   1.0,   1.0,  0.3,  -0.3, -0.3,  -1.0,  0.0],  # polar_unsat
    [ 0.0,     0.3,   0.3,    0.3,   0.3,   0.3,  0.2,   0.1,  0.2,  -0.5,  0.0],  # mixed
    [ 0.0,     0.3,  -0.5,   -1.0,  -1.0,  -1.0, -0.3,  -0.5,  0.5,  -1.0,  0.0],  # buried
])
```

**主要差异：**
| 环境 | 原子类型 | 版本 A | 版本 B |
|------|----------|--------|--------|
| hydrophobic | halogen | **1.0** | **0.3** |
| hydrophobic | unknown | -0.5 | 0.0 |
| polar_unsat | C_sp3 | -0.5 | -0.3 |
| polar_unsat | C_aromatic | -0.5 | -0.3 |
| polar_unsat | halogen | -0.5 | -0.3 |
| mixed | 所有 | **0.5** | **0.3** |
| mixed | S | 0.5 | 0.2 |
| mixed | halogen | 0.5 | 0.2 |

**两阶段生成使用版本 A**（来自 `latent_guidance.py`）。版本 B 是 `analytic_esfield.py` 的遗留代码。

**计划问题：** 矩阵应与哪个表（论文中的表 10）匹配？需要澄清。

---

### 3.3 原子类型概率

- 从模型的 `h` 输出（原子特征）获取
- 自动检测逻辑（例如，latent_guidance.py 第 427-432 行）：
  ```python
  if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum, ones):
      atom_probs = h  # 已经是 softmax
  else:
      atom_probs = F.softmax(h, dim=-1)  # 应用 softmax
  ```
- **没有应用温度缩放**
- DrugFlow 的 `h` 通常是 logits（最终线性层的输出），因此应用 `F.softmax(h, dim=-1)`

---

## 4. 生成器模型集成

### 4.1 DrugFlow 引导插入

**第一阶段：** `two_stage_generation.py` 第 911-919 行
```python
rdmols, trajectories, _ = self.model.sample(
    protein_data, n_samples=n_samples_per_attempt,
    timesteps=timesteps, num_nodes=cfg.n_init_atoms,
    guide_log_prob=guide_fn,  # ← _Phase1GuideFn
)
```

**第二阶段：** `two_stage_generation.py` 第 1112-1120 行
```python
rdmols, trajectories, _ = self.model.sample(
    protein_data, n_samples=n_samples,
    timesteps=timesteps, num_nodes=full_mol_size,
    guide_log_prob=guide_fn,          # ← TwoStageGuideFn（基于能量）
    post_step_callback=post_step_callback,  # ← HardFix 或 KinematicAnchor
)
```

### 4.2 ODE 步数

- **DrugFlow 第一阶段：** 默认 `timesteps=50`（不是 100）
- **DrugFlow 第二阶段：** 默认 `timesteps=100`（不是 200）
- DrugFlow 的 simulate() 使用欧拉积分，`delta_t = (t_end - t_start) / timesteps`

### 4.3 KPE 诊断

在 `KinematicAnchorGuidance` 中实现（kinematic_anchor.py）：
- **E_ODE 计算**（第 255 行）：`kpe_ode_step = (delta_x_ode²).sum() / dt`
- **E_guide 计算**（第 258-260 行）：`kpe_guide_step = (delta_x_guide²).sum() / dt`
  - `delta_x_guide = delta_x_total - delta_x_ode`
- **KPE 比率**（第 361-366 行）：`kpe_guide_total / (kpe_ode_total + kpe_guide_total)`
- 通过 `get_kpe_summary()` 方法汇总
- 每步历史每 10 步存储一次

### 4.4 流匹配与扩散的差异

**DrugFlow（流匹配）：**
- 直接速度预测 `v_θ(x_t, t)`
- 引导应用为：`v_guided = v_θ - λ * ∇_x E_site`
- 欧拉积分：`x_{t+Δt} = x_t + Δt * v_guided`
- 每个欧拉步后调用 `post_step_callback`

**TargetDiff（扩散）：**
- 从 `x_t` 预测 `x̂_0`，然后计算后验均值
- 引导应用于 `x̂_0` 预测：`x̂_0' = x̂_0 - λ * ∇_x E_site`
- 然后标准 DDPM 后验采样
- 文件：`targetdiff_esfield_guide.py`（显示了 100 行）
- 步数：通常 500-1000（DDPM 惯例）

---

## 5. 数据源

### 5.1 六个 PDBbind 口袋

这 6 个口袋是：
| PDB ID | PDBbind 年份 | 位点地图文件 |
|--------|-------------|--------------|
| 3mfw | 2001-2010 | `experiments/targetdiff_replication/site_maps/3mfw_site_map.json` |
| 2gni | 2001-2010 | `experiments/targetdiff_replication/site_maps/2gni_site_map.json` |
| 2gqn | 2001-2010 | `experiments/targetdiff_replication/site_maps/2gqn_site_map.json` |
| 2jke | 2001-2010 | `experiments/targetdiff_replication/site_maps/2jke_site_map.json` |
| 6o4x | 2011-2019 | `experiments/targetdiff_replication/site_maps/6o4x_site_map.json` |
| 6phx | 2011-2019 | `experiments/targetdiff_replication/site_maps/6phx_site_map.json` |

**选择标准：** 这些口袋被选择是因为它们含有具有明确定义的高能水（HEW）位点的结晶水。位点地图通过 `build_crystal_water_sites.py` 构建，该脚本：
1. 从 PDB 解析结晶水
2. 将每个水分类为"稳定"（>2 个氢键）或"高能水"（≤1 个氢键，≥3 个疏水接触）或"埋藏"（到最近蛋白质的距离 < 2.5 Å）
3. 所有水都在口袋中心 10 Å 范围内

**能否找到更多口袋？** 可以——PDBbind 包含数千个带有结晶水的结构。`build_pdbbind_water_sites.py` 脚本可以处理更多。标准是：PDBbind 复合物在结合口袋中至少有 1 个 HEW 位点。

---

## 所需更改摘要（将在第二部分执行）

| # | 问题 | 当前值 | 所需值 | 优先级 |
|---|------|--------|--------|--------|
| 1 | λ_Phase1 | 0.5 | **5.0** | 高 |
| 2 | λ_max（第二阶段 kinematic） | 0.5 | **1.0** | 高 |
| 3 | 硬固定实现 | 每步重置 | **完全锁定** | 高 |
| 4 | 兼容性矩阵 | 两个版本 | **统一为版本 A** | 中 |
| 5 | 第一阶段尝试次数默认值 | 3 | **100**（按计划） | 中 |
| 6 | 第一阶段 ODE 步数 | 50 | 应为 100 | 低 |
| 7 | 缺失：merge_anchors_to_initial_molecule() | 未实现 | 第 2.3 部分 | 高 |
| 8 | 缺失：第一阶段统计模块 | 未实现 | 第 2.2 部分 | 高 |
| 9 | 缺失：第二阶段退化为单阶段 | 未实现 | 第 2.3 部分 | 高 |