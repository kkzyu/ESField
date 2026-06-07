# ESField: Hydration-Site-Aware Molecular Generation for Structure-Based Drug Design

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-mechanism%20diagnosed-orange.svg)]()

**ESField** 研究结构基 3D 分子生成中的一个尚未被充分刻画的问题：现有 pocket-conditioned generator 能够生成空间上适配蛋白口袋、在常规质量指标上可接受的分子，但并不显式识别或利用口袋中的水合机会位点。尤其对于可能具有不利局部环境的保留水，如果新生成分子能够以化学合理的局部结构利用这些位点，理论上可能形成更有利的结合态候选。

本项目聚焦于 **bound-state molecular design**：研究如何将水合位点信息转化为对生成分子有意义的设计目标。项目并不模拟 ligand 从溶液进入口袋、排水并最终结合的完整动力学路径，也不将当前指标直接等同于真实结合自由能或临床活性。

> **当前研究状态**：项目已完成从 POSU 指标诊断、learned potential 诊断到 analytic coordinate guidance 验证的完整机制排查。实验表明：**仅依靠推断期 coordinate-only gradient，无法稳定诱导 DrugFlow 生成新的候选高能水占据结构。** 当前得到的是一条明确的 NO-GO 结论与控制层级诊断，而非已经验证成功的生成方法。

---

## 1. 研究问题

### 1.1 现有 3D 生成模型缺少显式的水合机会位点目标

DrugFlow、TargetDiff、DiffSBDD 等结构基 3D 生成模型通常学习：

```text
p(ligand | protein pocket)
```

此类模型能够学习蛋白-配体复合物的统计分布、形状互补与局部几何关系，但并不意味着模型会主动区分：

- 某个局部保留水是否可能成为可利用的设计机会；
- 哪种原子或局部取代基适合进入该区域；
- 稳定水应被保留、桥接还是补偿性替代；
- 疏水空腔或水合位点利用是否可能带来额外结合收益。

因此，一个生成分子即便在 validity、QED、SA 或 docking score 上表现尚可，也可能忽略口袋中具有药物设计价值的局部水合环境。

### 1.2 常规 docking / quality 指标不能直接评价水合位点利用

Vina、Gnina、QED、SA、validity、PoseBusters 等指标适合衡量分子整体质量、构象合法性或粗粒度结合倾向，但无法单独说明：

- 分子是否利用了某个特定候选水合位点；
- 正确类型原子是否进入了目标区域；
- 某个保留水是否应被替代；
- 所谓“位点利用”是否真的产生了水释放自由能收益。

因此，本项目需要独立的 site-level evaluation，用于描述生成分子对局部水合机会的利用情况；但该评价不能被直接等同于真实 binding affinity。

### 1.3 “靠近水位点”与“替换高能水”不是同一个物理事件

这是本项目在诊断过程中识别出的核心概念问题。

早期设想中，ligand atom 靠近 HEW site 被近似理解为 water displacement 的正信号。后续分析表明：

```text
atom 靠近 site ≠ atom 替换了该水
atom 替换了该水 ≠ 系统结合自由能一定下降
结合自由能下降 ≠ 只由 atom-site distance 决定
```

严格意义上的水替换是一个反事实自由能事件：需要知道未占据/水合状态下该位置是否存在水、该水的热力学性质、配体结合后该水是否离开、替代基是否补偿其原有相互作用，以及整体自由能代价与收益。

当前数据主要来自 bound complex，无法直接观察“某个原子确实替换了某个高能水并因此提高亲和力”。因此，使用 atom-site proximity 直接构造 learned potential 的正负样本，天然存在监督语义不足的问题。

---

## 2. 研究动机与目标边界

本项目的核心动机不是简单让原子靠近空间中的某个点，也不是只降低 clash rate，而是：

> **探索水合位点中包含的局部物理化学信息，能否转化为结构基 3D 分子生成中的有效设计约束，从而生成更可能形成有利结合态的候选分子。**

理想的设计逻辑为：

```text
可靠的水合机会位点
    + 合理的局部化学结构
    + 无严重蛋白碰撞与分子畸变
    + 整体质量与结合 proxy 不恶化
    → 更有利的 bound-state candidate
```

项目当前明确不作以下过度声明：

- 不将 rule-based water site 直接称为已验证的高能水自由能位点；
- 不将 POSU / HEWU 直接视为 binding free energy；
- 不声称当前生成结果能够预测临床疗效；
- 不声称已经模拟了真实水替换过程或结合动力学。

---

## 3. 数据语义：当前位点是什么，尚未证明什么

### 3.1 当前 site map 的来源

当前 HEW 候选位点全部来自 **holo crystal complex 中仍然保留的 crystal waters**，并通过规则进行筛选：

```text
hbond_count ≤ 1 OR hydrophobic_contact ≥ 3
```

因此，更准确的命名应是：

```text
candidate unfavorable retained water
candidate displacement site
rule-based candidate HEW
```

它们表示当前结合态中仍存在、但局部环境可能不利、值得进一步研究的候选水位点。

### 3.2 当前位点不是 displacement-positive gold label

Native ligand 到 candidate HEW 的分析结果为：

| 指标 | 数值 |
|---|---:|
| 总 atom–HEW pairs | 2334 |
| \(d < 2.0\) Å | 0 / 2334 |
| \(d < 3.0\) Å | 9 / 2334 |
| 平均距离 | 9.55 Å |

这一结果说明：

> **当前 native ligand 从未直接占据这些 candidate HEW 位点。**

因此，这些位点不能被解释为“reference ligand 已经验证有效的水替换位点”。它们更像是当前 ligand 尚未利用的潜在局部优化机会。当前研究能够评估生成分子是否向这些候选位点产生占据倾向，但尚不能证明这种占据必然降低结合自由能。

---

## 4. 初始方法框架与模块状态

项目最初采用以下四步框架：

```text
Step 1: 位点检测
Step 2: 位点-原子兼容性 / 势能构造
Step 3: 推断期 gradient guidance
Step 4: POSU / HEWU / quality 评估
```

经过系列实验后，各模块当前状态如下：

| 模块 | 初始目标 | 当前认识 |
|---|---|---|
| Step 1 位点检测 | 找到 HEW / SW / HC | 已得到 rule-based candidate water sites，但并非经热力学验证的 displacement-positive 位点 |
| Step 2 势能构造 | 学习或显式描述 atom-site 利用收益 | learned potential 与 analytic reward 均已被诊断；水替换语义不能由 proximity 简单替代 |
| Step 3 推断期 guidance | 通过坐标梯度促使分子利用候选水位点 | coordinate-only route 已被 actionable-pocket validation 判定为 NO-GO |
| Step 4 评价指标 | 衡量位点利用和基本质量 | POSU-v2.1/HEWU 可作为 site-level 辅助诊断；不能等同于 affinity 或 displacement free energy |

---

## 5. 研究历程与关键发现

### 5.1 选择 DrugFlow 作为 base generator

项目早期考察了若干 structure-based 3D generator。在当前代码集成与生成测试中，DrugFlow 在有效生成率、常规分子质量以及推断期可操作性方面更适合作为 base generator，因此后续实验均以 DrugFlow 为基础。

DrugFlow 在本项目中的角色是：

```text
强结构基 3D 分子生成 backbone
而不是显式 hydration-aware generator
```

项目要检验的是：在一个具备合理基础生成能力的模型上，水合位点信息能否进一步提供新的设计信号。

---

### 5.2 POSU-v1：首次暴露 site identity 未被指标可靠反映

扩展到 PDBbind water-site 数据后，原始 POSU-v1 在 correct / random / shuffled site map 下差异很小：

```text
correct site map ≈ random site map ≈ shuffled site map
```

这并不能直接证明 site 没有意义，而是首先说明：**POSU-v1 对 atom type 与 site type 的兼容关系不敏感。**

诊断出的主要问题包括：

- HEW compatible 规则过宽，随机位点也容易得分；
- SW 被简单处理为“不可破坏”，会误伤真实配体可能存在的补偿性替代；
- HEW reward 与 SW penalty 在 composite POSU 中互相抵消；
- max/top-k 聚合可能放大偶然接近；
- shuffled-type 在保留真实坐标时可能产生不自然的高分。

---

### 5.3 POSU-v2.1：修复评价指标的部分结构性缺陷

为改善 site-level evaluation，POSU-v2.1 引入：

- HEW environment classification：hydrophobic / polar_unsatisfied / mixed / buried；
- mixed HEW 降权；
- SWP / SWBR / SWCR / SWDP 分解；
- SW disruption hard threshold；
- HEW 主导、SW 约束、HC 辅助的加权策略。

关键结果：

| 诊断项 | POSU-v1 | POSU-v2.1 |
|---|---:|---:|
| Native correct > random PASS | 6/20 | 12/20 |
| SW 反向问题 | -0.086 | -0.038 |
| HEWU signal | +0.116 | +0.103 |

结论：

> POSU-v2.1 比 v1 更适合用于 site-level diagnostic，但仍不是真实结合自由能或水替换收益的代理金标准。

---

### 5.4 v4 learned potential：模型学到距离捷径，而非可靠兼容性

早期 learned potential 试图从 atom type、site type 与距离中学习可用于 guidance 的能量函数。v4 的 ordinary AUC 看似良好，但在控制距离分布后性能明显下降；force matrix 还显示 incompatible atom-site pairs 也会被吸引。

实验表现为：

- distance-matched AUC 仅约 0.701；
- 关键距离区间的类型判别能力较弱；
- 不兼容 pair 同样受到吸引；
- generation 后 HEW-compatible atom 反而可能远离目标 site。

结论：

> v4 主要学习到了“距离近更像正样本”的 shortcut，而不是能够支撑生成引导的 atom-site compatibility。

---

### 5.5 v5 learned potential：pair-level 修复成功，但 generation-level 引导仍不成立

v5 针对 v4 引入：

- same-distance wrong-type negatives；
- hand-crafted distance well；
- MLP 仅学习 \(\alpha/\beta\) compatibility coefficients；
- distance-matched validation。

离线诊断显示 v5 显著修复了 pair-level 类型学习：

| 指标 | v4 | v5 |
|---|---:|---:|
| Distance-matched AUC | 0.701 | 0.989 |
| 2.5–3.5 Å AUC | 0.657 | 1.000 |
| HEW \(\alpha\)(pos/neg) | 约 1:1 | 3.3:1 |

但是，在真实 DrugFlow generation 中：

- POSU / HEWU 改善幅度微弱；
- wrong-pocket 区分接近噪声；
- 部分 pocket 没有响应或出现反向表现；
- sum_norm 虽能改善 reranking 相关性，却不能改善实际 guidance。

结论：

```text
pair-level compatibility ≠ molecule-level ranking
molecule-level ranking ≠ generation-time gradient
好的 scoring energy ≠ 好的 guidance energy
```

v5 证明了 learned potential 可以在局部 pair 判别上变好，但没有证明其梯度能够稳定改变生成分布。

---

### 5.6 概念修正：proximity 不能作为 replacement 的等价标签

在 v5 之后，项目识别出更根本的问题：早期方法将“原子靠近 HEW site”作为“替换高能水”的近似监督，但当前数据并不提供水替换的真实标签。

尤其是在确认当前位点均来自 holo complex retained waters，且 native ligand 从未直接占据 candidate HEW 后，原有正样本逻辑不再成立：

```text
observed ligand proximity
    ≠ verified water displacement
    ≠ verified local free-energy gain
```

这一发现重新界定了本项目能够证明的范围：

- 当前可以研究 candidate water-site occupancy / utilization；
- 当前不能直接将其宣称为真实 water displacement free-energy optimization；
- 若要支撑更强自由能叙事，需要额外的水热力学校准或真实 displacement positive-control。

---

### 5.7 v6-D：解析 displacement reward 的第一次尝试

为避免 learned potential 的弱标签与黑盒梯度问题，项目实现了 Analytic ESField v6-D：

```text
displacement occupancy reward
+ atom-site compatibility matrix
+ wrong-type penalty
+ protein clash penalty
+ overfill penalty
```

v6-D 的离线 toy tests 证明：

- 兼容原子可被正确吸引；
- 不兼容/带电原子可被抑制或惩罚；
- protein clash 与 overfill 可受到保护；
- analytic field 对坐标可微；
- random matrix 与正确矩阵在 toy landscape 上行为不同。

然而，5-pocket generation test 中 v6-D 与 baseline 几乎无差异。进一步诊断发现，初始 occupancy reward 以水中心 \(d=0\) 为目标，但生成分子中的最近原子通常距离 candidate HEW 约 3.5–7 Å，导致窄 Gaussian 在真实轨迹上的梯度近乎不可用。

结论：

> v6-D 的物理目标比 v5 更清楚，但仅有窄 occupancy reward 不能从当前生成分布中捕获原子进入候选水位点。

---

### 5.8 v6-D.2：Capture-to-Displacement 解析场仍未解决生成控制问题

v6-D.2 在 v6-D 基础上引入双尺度解析奖励：

```text
capture term：在较远距离提供向 candidate HEW 的吸引梯度
occupancy term：奖励进入水中心邻域的直接占据
protection terms：抑制错误类型、碰撞与过度占位
```

首先在 2clh 上进行快速验证。2clh 的单步梯度响应表明：

- ESField 梯度存在且方向合理；
- 但该 pocket 的 baseline 已经接近 candidate HEW，并存在 direct occupancy；
- 因此 2clh 更适合作为 ceiling/stability control，而不适合作为增益验证对象。

随后对 baseline 口袋进行机会分层：

| 类别 | 口袋 | 含义 |
|---|---|---|
| B: Actionable | 2gni, 3mfw, 6o4x | 基线尚未充分占据但候选 site 具有可测试空间 |
| A: Ceiling | 1sle, 2clh, 3b28 | baseline 已有占据，不适合验证新增收益 |
| C: Too far | 4pax | 候选位点过远 |
| D: Ambiguous | 5g60 | buried / 质量异常 |
| No candidate HEW | 2jkr, 2rox, 2wgi, 3ohi, 3tcp, 4bis | 不适用于当前 displacement 测试 |
| Need baseline | 1tyn, 2aod, 3r01, 6ayn, 6hvh, 6hy2 | 尚待补充 |

在 B 类 actionable pockets 上进行正式验证后，v6-D.2 得到 NO-GO 结果：

#### 2gni

| 指标 | Baseline | v6-D.2 \(\lambda=1.0\) | Random | Wrong |
|---|---:|---:|---:|---:|
| DirectOcc | 0/60 | 2/60 | 1/57 | 2/54 |
| MinCompatD | 3.93 Å | 3.78 Å | 4.03 Å | 3.87 Å |
| SoftOcc | 0.022 | 0.023 | 0.022 | 0.036 |

表现：correct guidance 有极弱距离变化，但 wrong-pocket 可达到相近或更高 occupancy，无法证明 site-specific displacement。

#### 6o4x

| 指标 | Baseline | v6-D.2 \(\lambda=1.0\) | Random | Wrong |
|---|---:|---:|---:|---:|
| DirectOcc | 0/51 | 0/48 | 1/54 | 1/60 |
| MinCompatD | 4.37 Å | 4.32 Å | 4.22 Å | 4.19 Å |
| QED | 0.558 | 0.419 | 0.588 | 0.351 |

表现：correct guidance 未产生 direct occupancy，random / wrong 反而更好，并伴随明显 QED 下降。

#### 3mfw

| 指标 | Baseline | v6-D.2 \(\lambda=1.0\) |
|---|---:|---:|
| DirectOcc | 0/20 | 0/20 |
| MinCompatD | 5.04 Å | 5.03 Å |
| SoftOcc | 0.000 | 0.000 |

表现：完全无响应。

最终结论：

> **v6-D.2 能提供方向合理的局部坐标梯度，但不能在 DrugFlow 的真实生成轨迹中稳定诱导新的 candidate HEW occupancy。**

### 5.9 v7 — Two-Stage Hierarchical Latent Guidance（当前方向）

基于 v6‑D.2 的 NO‑GO 结论，v7 从根本上改变策略：不再试图通过 coordinate‑only gradient 微调已有原子的位置，而是显式地将生成过程拆分为两个阶段，以实现在 HEW 候选位点处的 **拓扑层面控制**（新增原子、选择兼容类型、生长取代基）。

**Phase 1 — OCCUPY（占位）**
- 用小片段（默认 4 个原子）+ 强 site‑compatibility guidance（λ=0.5）生成初始锚点
- 目标：至少一个兼容类型原子进入 candidate HEW 周围 2.5Å 内
- 使用 `SiteCompatibilityEnergy` —— 硬编码兼容性矩阵（4 种 HEW 环境 × 11 种原子类型），基于规则而非学习
- 采用 **KTS (Kinetic Trajectory Shaping)** 进行时变速度缩放：early boost (α₀=0.01) 促进拓扑探索，late exponential damp (β₀=0.01) 避免过度约束

**Phase 2 — CONNECT（连接）**
- 固定 Phase 1 得到的锚原子（harmonic restraint, k=10.0 kcal/mol/Å²）
- 以较弱的 guidance（λ=0.1）继续采样完整分子
- 可选：TypeGuidanceBias 额外偏置原子类型 logits 以改善兼容性

**v7 与 v6‑D.2 的关键区别**

| 维度 | v6‑D.2 (NO‑GO) | v7 (当前) |
|---|---|---|
| 控制层级 | Coordinate‑only nudging | Topology‑level: 新增原子 + 类型选择 |
| 位点利用方式 | 推动已有原子靠近 site | 在 site 处显式生长新片段 |
| 能量函数 | Capture + occupancy 双高斯 | 单一 compatibility‑weighted Gaussian (σ=3.0Å) |
| 时间策略 | 分段 capture/occ 权重 | KTS: early boost + late damp |
| 生成流程 | 单次采样 | Phase1 (occupy) → Phase2 (connect) |
| 需要训练 | 否 | 否（全部规则硬编码） |

**已完成的代码与测试**

| 模块 | 文件 | 测试 |
|---|---|---|
| SiteCompatibilityEnergy + 引导函数 | `src/guidance/latent_guidance.py` | 39 tests |
| KTS scheduler | `src/guidance/kinetic_trajectory_shaping.py` | (含于 latent + two_stage) |
| Two-stage orchestrator | `src/guidance/two_stage_generation.py` | 28 tests |
| Site occupancy 指标 | `src/evaluation/site_occupancy.py` | (含于 existing posu tests) |
| CLI 集成 | `scripts/run_v7_two_stage.py` | 待 GPU 运行 |
| 配置 | `configs/v7_config.yaml` | — |

**当前实验状态**

- ✅ v7 全部代码编写完成，与现有 ESField 目录完全兼容
- ✅ 67 个新增单元测试全通过（兼容性矩阵、能量梯度、KTS 调度、Phase1 诊断、Phase2 复合引导、RDKit 转换）
- ⬜ GPU 运行验证 — 需要在 actionable pockets (2gni, 3mfw, 6o4x) 上运行 Phase 1 + Phase 2
- ⬜ 与 v6‑D.2 baseline 对比 direct occupancy rate、best compatible distance、POSU/HEWU
- ⬜ Phase 1 成功率统计（多少 attempt 能成功占位）

---

## 6. 当前已确认的核心问题

### 6.1 位点标签语义仍不足以支持强自由能结论

当前位点是通过 holo retained crystal water 的局部规则识别得到的 candidate sites，而不是具有已知 displacement free-energy gain 的 gold labels。

因此，即使某个分子进入 candidate site，也只能首先解释为：

```text
candidate hydration-site occupancy / utilization
```

不能直接解释为：

```text
confirmed high-energy water displacement
or improved binding free energy
```

### 6.2 POSU / HEWU 是诊断指标，不是最终亲和力指标

POSU-v2.1 修复了 v1 的若干结构性缺陷，可用于分析 site-level utilization；但当前数据无法证明其与真实 binding affinity 或 water displacement free energy 单调对应。

### 6.3 Learned potential 无法解决不可观测 replacement label 问题

v4 学到 distance shortcut；v5 虽然修复了 pair-level compatibility，却没有形成可靠的 generation-time gradient。原因不仅在于训练技巧，更在于：

```text
水替换事件本身无法由当前 proximity-based pair labels 直接监督。
```

### 6.4 Analytic displacement field 仍受控制变量限制

v6-D 与 v6-D.2 将能量定义变得更透明，但 generation 实验表明：

- 局部梯度存在；
- 正确矩阵在 toy test 中行为合理；
- 在真实生成中，correct 未稳定优于 random / wrong；
- 提高局部吸引可能损害分子质量；
- 远距离 candidate site 无法通过轻量坐标修改被利用。

因此，问题不再主要是势能函数是否可解释，而是：

> **coordinate-only gradient 作用于错误的控制层级。**

### 6.5 Candidate HEW 利用往往需要局部拓扑变化

当前 candidate HEW 是 native ligand 未直接占据的潜在机会空间。要真正利用此类位点，生成分子往往需要：

- 新增或改变局部取代基；
- 调整原子类型；
- 改变键连接或分支方向；
- 同时维持几何合法性与分子质量。

单纯推动已有 atom coordinates 只能进行轻量 pose adjustment，无法稳定完成新的局部结构设计。

---

## 7. 已证实与尚未证实的结论

### 已证实

| 结论 | 证据 |
|---|---|
| POSU-v1 对 site identity 不敏感 | correct/random/shuffled 结果接近 |
| POSU-v2.1 改善了 site-level diagnosis | native correct>random 通过率提升、SW 反向减弱 |
| v4 存在 distance shortcut | distance-matched AUC 降低、非特异性吸引 |
| v5 能修复 pair-level compatibility | distance-matched AUC 与关键区间 AUC 显著提高 |
| v5 的 pair-level 成功不能保证 generation 成功 | generation 效果弱且对照区分不足 |
| 当前 site 来自 holo retained waters | site map 来源分析 |
| Native ligand 未直接占据 candidate HEW | 0/2334 atom-site pairs within 2 Å |
| v6-D 窄 occupancy reward 在真实轨迹中梯度不足 | 5-pocket 中与 baseline 接近 |
| v6-D.2 coordinate-only guidance 为 NO-GO | actionable-pocket validation 未达标准且对照不支持 |
| HEW 利用需求涉及 topology-aware control | correct coordinate nudging 不能稳定产生新 occupancy |
| v7 两阶段生成代码全部完成 | 67 新增单元测试全通过，与现有代码兼容 |
| v7 硬编码兼容性矩阵可在 toy test 中正确吸引/排斥原子 | 梯度方向、soft probs、multi-site、KTS 全部验证 |
| **v7.1 在 7/10 口袋实现 DirectOcc > 0%** | v6-D.2 baseline = 0%，v7.1 提升至 4-36% |
| **消融：λ=5.0 最佳，硬覆盖与软约束差异不显著** | 各组件贡献均衡，类型策略影响极小 |
| **v7.1 多样性高于 baseline** | Vendi +27-45%，引导不牺牲多样性 |
| **占据组 Vina 得分不优于未占据组** | 3mfw p=0.067 趋势反向（占据组更差）|

### 尚未证实

| 命题 | 当前状态 |
|---|---|
| Rule-based candidate HEW 确实具有正 displacement free-energy benefit | 未证实 |
| 占据 candidate HEW 会降低结合自由能 | 未证实 |
| POSU / HEWU 能预测亲和力提升 | 未证实 |
| 任一 hydration-aware generative method 已能稳定改善 candidate HEW utilization | 未证实 |
| 生成分子具有更好的实验活性或临床意义 | 未证实 |

---

## 8. 当前项目状态

```text
Research motivation:
  Hydration-site-aware molecular design remains scientifically meaningful.

Metric diagnosis:
  POSU-v2.1 retained as a site-level diagnostic metric, not an affinity proxy.

Site semantics:
  Current HEW sites are rule-based candidate unfavorable retained waters,
  not validated displacement-positive sites.

Learned potential:
  v4/v5 diagnosed; pairwise potential does not provide stable generation guidance.

Analytic coordinate guidance:
  v6-D and v6-D.2 implemented and tested.
  Actionable-pocket validation result: NO-GO.

Two-stage topology control (v7):
  Code complete, 67 new tests passing.
  Phase1 (Occupy) + Phase2 (Connect) with KTS time-shaping.
  Hard-coded compatibility matrix, no training required.
  Awaiting GPU validation on actionable pockets.

Current central finding:
  Candidate HEW utilization likely requires control over local topology,
  atom types, bond types, or substituent growth — not just coordinate
  nudging.  v7 is the first ESField method designed for this regime.
```

---

## 9. 当前代码与实验资产

### 已实现模块

| 模块 | 状态 | 说明 |
|---|---|---|
| DrugFlow baseline integration | 已完成 | 生成 backbone |
| Site-map construction | 已完成 | rule-based candidate water sites / cavity sites |
| POSU-v1 | 已完成 | 早期指标，保留作历史对照 |
| POSU-v2.1 / HEWU / SWScore | 已完成 | site-level diagnosis |
| v4 learned potential | 已完成 | distance shortcut 诊断对象 |
| v5 learned potential | 已完成 | pair-level 修复但 generation 不稳定 |
| v6-D analytic field | 已完成 | narrow occupancy reward |
| v6-D.2 capture-to-displacement field | 已完成 | coordinate-only NO-GO 测试 |
| **v7 latent guidance** | **代码完成** | SiteCompatibilityEnergy + KTS + two-stage |
| **v7 two-stage generation** | **代码完成** | Phase1 Occupy + Phase2 Connect |
| **v7 site occupancy metrics** | **代码完成** | direct_occupancy_rate, best_compatible_distance |
| Actionable-pocket baseline analysis | 部分完成 | 14 pockets 已分层，6 pockets 待补 baseline |

### 关键实验目录

```text
experiments/pdbbind_water_sites/
├── site_maps/
├── v4_diagnosis/
├── v5_mechanism_test/
├── v5_sum_norm_test/
├── v6_displacement_test/
├── v6d2_quick_test/
├── v6d2_actionable_validation/
└── v7/                              # ★ 新增
    ├── phase1_occupy_test/          # Phase 1 成功率测试
    └── two_stage_full/              # 完整两阶段生成结果
```

---

## 10. 项目结构

```text
ESField/
├── src/
│   ├── site_detection/                  # candidate hydration-site detection
│   ├── data/                            # pair/sample construction
│   ├── models/
│   │   ├── potential_network.py         # v4/v5 learned potential
│   │   └── analytic_esfield.py          # v6-D / v6-D.2 analytic field
│   ├── evaluation/                      # POSU-v2.1 / HEWU / SWScore
│   │   └── site_occupancy.py            # ★ v7: direct occupancy metrics
│   ├── guidance/                        # inference-time guidance modules
│   │   ├── latent_guidance.py           # ★ v7: SiteCompatibilityEnergy
│   │   ├── kinetic_trajectory_shaping.py  # ★ v7: KTS scheduler
│   │   └── two_stage_generation.py      # ★ v7: Phase1+Phase2 orchestrator
│   ├── training/                        # learned potential training
│   └── utils/
├── scripts/
│   ├── drugflow_esfield_guide.py
│   ├── run_v7_two_stage.py              # ★ v7: 两阶段生成 CLI
│   ├── build_pdbbind_water_sites.py
│   ├── build_v5_pairs.py
│   ├── train_potential_v5.py
│   ├── validate_v5_offline.py
│   ├── run_v5_mechanism_test.py
│   ├── run_sum_norm_experiment.py
│   ├── run_v6_displacement_test.py
│   ├── run_v6d2_quick_test.py
│   └── analyze_v6d2_actionable_validation.py
├── configs/
│   └── v7_config.yaml                   # ★ v7: 超参数配置
├── experiments/
├── docs/
│   ├── v6-D_实现与诊断报告_2026-05-21.md
│   └── v7_实现与设计报告_2026-06-04.md    # ★ v7: 实现报告
├── paper/
└── tests/
    ├── test_analytic_esfield.py          # 19 v6-D tests
    ├── test_latent_guidance.py           # ★ v7: 39 tests
    └── test_two_stage.py                 # ★ v7: 28 tests
```

---

## 11. 快速开始

### 环境

```bash
# Python 3.12, PyTorch 2.6+, CUDA 12.4
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs:$LD_LIBRARY_PATH"
```

### 单元测试

```bash
cd /root/ESField
PYTHONPATH=src python -m unittest discover -s tests
# 76 passed (9 original + 19 v6-D + 48 v7), 10 pre-existing errors in test_analytic_esfield.py
```

### 构建 candidate water site maps

```bash
PYTHONPATH=src python scripts/build_pdbbind_water_sites.py \
  --pdbbind-root /root/autodl-tmp/data/PDB/P-L \
  --output-dir experiments/pdbbind_water_sites \
  --n-pockets 100 \
  --max-atoms 2000
```

### 运行已有 diagnostic experiments

```bash
# v5 learned potential mechanism test
PYTHONPATH=src python scripts/run_v5_mechanism_test.py

# v6-D analytic displacement test
PYTHONPATH=src python scripts/run_v6_displacement_test.py

# v6-D.2 actionable-pocket validation
PYTHONPATH=src python scripts/run_v6d2_quick_test.py
```

---

## 12. Citation

```bibtex
@misc{esfield2026,
  title={Hydration-Site-Aware Molecular Generation for Structure-Based Drug Design: Diagnosing Coordinate-Only Guidance for Candidate Water-Site Utilization},
  author={ et al.},
  year={2026},
  note={Work in progress}
}
```

---

*Last updated: 2026-06-04 (最终实验完成)*
