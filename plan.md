Prompt：完善 Kinematic Anchor Guidance (KAG) 项目并重构论文

请你阅读@root/ESField和@root/ESField/papper_tex/main.pdf，完成以下四个部分的工作。注意，如果在阅读和执行过程中，有遇到未表述清楚的问题、细节模糊的方案、路径不清的资源，请及时询问我，不要着急直接开展下一步工作。（擅作主张地决定细节/调整步骤/更改方案的行为都是决不允许的）

第一部分：确认现有代码实现细节（请逐项检查并回答）

请阅读当前代码库（@root/ESField/script），回答以下问题。如果某项未实现或不明确，请指出并准备修改。

1. Phase 1 (Occupy) 的实现细节

· Phase 1 生成小片段时，使用的是 per‑atom 全梯度（∇_x E_site 每个原子独立）还是 CoM 投影梯度（先对梯度平均再统一加）？
  · 如果是 CoM 投影，请修改为 per‑atom 全梯度。
· Phase 1 的引导强度 λ_Phase1 是否设置为 5.0？是否使用了 KTS early boost（参数 τ_split 和衰减率）？
· Phase 1 的原子数固定为 4 吗？是否在每次运行中随机初始化？
· Phase 1 结束后，如何判定哪些原子是“锚点”？请给出代码中的距离阈值（如 2.5 Å）和化学兼容性得分阈值（如 0.3）。
· 锚点的信息（坐标、原子类型、对应的 HEW 位点 ID）如何保存和传递给 Phase 2？

2. Phase 2 (Connect) 的实现细节

· Phase 2 中是否实现了 CoM 投影？请展示计算 ∇_CoM E_site 并统一加到所有原子的代码片段。
· Phase 2 的时间调度 λ(t) 是否按照公式 (10) 实现？λ_max 是否为 1.0？衰减是否为 (1-t)^2？
· 硬固定 (hard‑fix) 基线是如何实现的？
  · 是每步重置锚点坐标到初始固定值（允许 ODE 和引导更新后强制改回）？
  · 还是完全锁定锚点坐标（不参与任何更新，也不受引导）？
  · 请明确。如果当前是“每步重置”，请改为“完全锁定”（即锚点坐标在 Phase 2 全程为常数），并重新运行 hard‑fix 基线以获得公平对比。

3. 势能场 E_site 的实现

· 公式 (1) 中的参数：σ 是否为 3.0？τ 是否为 10？是否所有 HEW 位点共享同一 σ？
· 兼容性矩阵 M 的数值是否与论文附录表 10 完全一致？请输出该矩阵的 Python 代码表示（4×11 的 numpy 数组）。
· 原子类型概率 h_{i,a} 是从模型的 softmax 输出中直接获取吗？是否经过了温度缩放？

4. 生成模型集成

· DrugFlow 和 TargetDiff 的采样循环中，引导步骤是如何插入的？请指出具体的文件和行号。
· ODE 步数 T 是多少？对于 DrugFlow，是否使用 100 步？对于 TargetDiff，是否使用 500-1000 步？
· KPE 诊断的计算是否已实现？请提供计算 E_ODE 和 E_guide 的代码位置。
· 对于流匹配模型和扩散模型分别有哪些不同的处理方式？

5. 数据来源
 ·  “3mfw, 2gni, 6o4x, 2jke, 2gqn, 6phx”这六个PDBbind口袋是如何被筛选出来的？是否能够筛选出更多类似的口袋？

请你先将对以上问题的回答写入到一份markdown文件中。

---

第二部分：方案细节调整（请修改代码或新增函数）

基于第一部分的检查，请执行以下修改/新增：

2.1 改进高能水利用的评估指标

编写一个独立的 Python 脚本 metrics_new.py，输入为生成分子的 SDF 文件（或 PDB 文件）和 HEW 位点信息（JSON 格式），输出一个汇总 JSON 文件，包含每个条件的统计量（均值、标准差、中位数），方便后续直接导入论文表格。同时输出以下指标记录在JSON文件中：

1. 质心到 HEW 位点的距离
   · 对每个分子计算质心 (x,y,z)
   · 对每个 HEW 位点计算欧氏距离，报告 最小距离 和 平均距离（对所有位点平均）
   · 输出为 CSV：mol_id, min_dist_centroid, avg_dist_centroid
2. 连续占用得分（Continuous Occupancy Score, COS）
   · 对每个 HEW 位点 k 计算：
          COS_k = max_i [ exp(-d_ik^2/(2σ^2)) * Σ_a h_i,a * M_{e_k,a} ]
          其中 σ=1.5 Å，d_ik 是原子 i 到位点 k 的距离，h_i,a 是原子类型概率（若模型输出 one‑hot 则直接使用）
   · 最后报告所有 HEW 位点的 平均 COS 和 最大 COS（即最佳匹配位点的得分）
   · 输出 CSV：mol_id, avg_COS, max_COS
3. 势能场值 E_site
   · 直接使用公式 (1) 计算每个分子的 E_site（不需要最小化）
   · 输出 CSV：mol_id, E_site
4. 帕累托前沿分析
   · 对每个条件（unguided, hard‑fix, KAG），以 Strain 为 x 轴，以 -min_dist_centroid 或 avg_COS 为 y 轴绘制散点图
   · 计算每个条件的帕累托边界（非支配点数量或比例）
   · 保存图表为 pareto_strain_vs_proximity.png

2.2 Phase 1 统计函数

在 Phase 1 运行后，增加一个统计模块，输出以下信息（保存为 phase1_stats.csv）：

· 运行总次数（尝试100次，每次重试都从新的随机噪声开始）
· 成功产生至少一个锚点的运行次数及比例
· 成功运行中，平均每个蛋白质分子产生的锚点数（及其标准差）
· 每个 HEW 位点被至少一个锚点覆盖的比例（跨所有成功运行）
· 所有成功运行中，每个分子的 E_site 最终值
· 每个分子最终是否使用了锚点（即成功次数占尝试次数的比例）。如果退化到 single-stage，应在 Phase 1 统计中标记该分子为“无锚点”。

2.3 确保 Phase 1 到 Phase 2 的衔接正确

· 编写函数 merge_anchors_to_initial_molecule(phase1_fragment, anchor_list)，确保 Phase 2 的初始分子是 Phase 1 生成的整个小分子（所有原子和键），而不是仅锚点原子。
· 运行phase1，若某次尝试产生了至少一个锚点→则记录该次生成的整个小片段，跳出循环，进入phase2；若所有尝试都失败→则退化为single-stage KAG（无锚点）。phase2以phase1得到的小片段（若有锚点）或纯噪声（若退化为single-stage）为初始分子，生长为完整分子。
· 每个完整分子都对应一次独立的phase1尝试序列，不同分子之间不共享phase1的结果。

---

第三部分：需要补做的实验（包括图表制作）

请按照以下实验清单执行，每个实验需输出指定图表。所有新生成的分子和中间数据请整理到 results/ 目录下。对于每个实验的结果都分别创建一个子文件夹进行存储，每完成一个实验就生成一次详细的报告。

3.1 Phase 1 锚点生成能力评估

· 模型：DrugFlow
· 口袋：全部 6 个 PDBbind 口袋（3mfw, 2gni, 6o4x, 2jke, 2gqn, 6phx，若这六个口袋在phase1阶段都为成功生成锚点的话，则在数据库中查找更多更简单的口袋进行尝试）
· 运行次数：每个口袋 100 次 Phase 1（使用 per‑atom 全梯度，λ=5.0）
· 输出图表：
  · 条形图：每个口袋的锚点成功率（成功运行比例）
  · 热图：每个 HEW 位点被覆盖的比例（行=口袋，列=位点编号）
  · 表格：平均锚点数、锚点类型的分布（例如 Csp3 占比等）

3.2 新指标下的结果重分析

· 输入：你已经生成的 DrugFlow 分子（无引导、硬固定、KAG，每个口袋 50 个分子）
· 计算：使用第二部分编写的 metrics_new.py，计算每个分子的质心距离、COS、E_site
· 输出图表：
  · 箱线图（或小提琴图）：比较三个条件的 min_dist_centroid（y 轴为距离，越小越好）
  · 累积分布图（CDF）：比较三个条件的 avg_COS
  · 帕累托散点图（每个口袋单独一张，以及所有口袋合并一张）：x=Strain，y=avg_COS，颜色区分条件
  · 表格：每个口袋下三个条件的平均质心距离、平均 COS、平均 E_site，并附 Wilcoxon 检验 p 值（KAG vs 无引导）

3.3 硬固定基线真锁定对比

· 重新运行 hard‑fix 基线，使用“完全锁定”锚点坐标的实现（不参与任何更新）。
· 对比两种硬固定实现（原“每步重置” vs 新“完全锁定”）的 strain 和质心距离。
· 输出箱线图：strain 分布对比。如果新实现的 strain 更高，说明原 hard‑fix 低估了应变，更凸显 KAG 优势。

3.4 跨架构验证：在 扩散模型[DecompDiff](https://github.com/bytedance/DecompDiff) （本地位置@/root/baselines/DecompDiff，ckpt位置@/root/autodl-tmp/checkpoints/DecompDiff）上测试 KAG
· 需要做的实验和drugflow完全一致。

3.5 药效团约束场景的扩展（已在 SI 中，但过于简单，需重做）

· 选择已有共晶配体的口袋（与HEW实验一致，方便对比），从参考配体中提取药效团特征点（11个点，含HBD,HBA,疏水，芳环等等）。保留这些点作为约束目标。
· 对于所有引导方法（硬固定、全梯度、KAG），使用同一组“理想锚点”：即从参考配体中提取的、与药效团点匹配的原子坐标和类型。这些锚点不作为初始分子的一部分，而是作为外部约束目标：引导力会将分子中的原子拉向这些锚点位置（兼容性由药效团兼容矩阵决定）。
· 对于每个药效团点p，定义其目标原子类型集合，并分配一个虚拟锚点坐标。在phase2中（无需phase1），直接对完整分子施加引导：E_pharm的计算类似于E_site，讲药效团视为“位点”。
· 使用drugflow作为生成器，每个条件生成50个分子
· 计算接近度指标（质心到最近药效团的距离，连续药效团得分，平均距离到所有药效团点），物理合理性（strain、clash、KPE比值），成药性（QED,SA），帕累托分析（strain vs 质心距离）

3.6 推理时间记录

· 在 DrugFlow 上，对 3mfw 口袋，随机抽取 10 个分子，记录：
  · Phase 1 平均耗时（秒）
  · Phase 2 平均耗时（秒）
  · 硬固定平均耗时（秒）
  · 无引导平均耗时（秒）
· 输出表格：平均时间 (ms/molecule)，并在论文实验设置中引用。

---

第四部分：根据结果调整论文行文逻辑（大纲）

以下是论文结构调整大纲，严格按照 二级标题 + 段落 方式（不使用三级标题）。每个章节明确需要展示哪些图表（引用图表编号，请根据实际生成后填充）。相比于原论文，我们需要按照以下行文逻辑进行调整和完善。

1. Introduction

· 内容：暂不做更改
· 图表：无

2. Preliminaries

· 2.1 High‑Energy Water Sites：定义、分类（hydrophobic/polar‑unsatisfied/mixed/buried）、热力学价值。
· 2.2 Kinetic Path Energy (KPE)：解释 KPE 公式，用于诊断引导注入的动能。
· 2.3 Site‑Compatibility Energy：给出 E_site 公式 (1)，解释兼容性矩阵 M 和梯度计算。
· 图表：无

3. Method
对问题定义和框架描述的部分还是保持不变。
· 3.1 Two‑Stage Generation Framework：概述 Phase 1 和 Phase 2 的分工，以及为什么要两阶段。
· 3.2 Phase 1: Anchor Seeding：使用 per‑atom 全梯度、强引导、小片段。说明锚点定义和统计方法。
· 3.3 Phase 2: Kinematic Anchor Guidance：CoM 投影、零应变证明（Theorem 1）、时间调度。
· 3.4 Extension to Pharmacophore Constraints：说明KAG不依赖于HEW的特定物理意义。只需要一组空间位点和一个化学兼容性矩阵M即可实现。这部分以药效团约束为例：定义药效团特征点，构建对于的兼容性矩阵。在Phase2中直接使用相同CoM投影引导。说明背景和意义，并描述使用的方法。
· 图表：重绘的 Figure 1（Overview，暂且用占位图进行占位）

4. Experiments

· 4.1 Experimental Setup：数据集、基线（Unguided, Hard‑Fix, Full Gradient，舍弃掉原来的Lai+SoftFix, BADGER）、模型（DrugFlow, decompdiff）、指标（传统指标 + 新引入的质心距离、COS）。表格：参数设置。
· 4.2 Results on Flow‑Matching Backbone (DrugFlow)
	  · Phase 1 anchor seeding success：展示 条形图和热图。解释：成功率高/低的口袋对应后续 HEW 占用潜力。
	  · HEW proximity and Pareto improvement：展示质心距离箱线图（图 Y）、COS 累积分布图（图 Z）、帕累托散点图（图 W）。表格：三个条件平均指标 + p 值。
	  · Physical plausibility and drug‑likeness：展示 strain、clash、vina、QED、SA 的表格。强调 KAG 在 strain 和 vina 方差上的优势。
· 4.3  Results  on Diffusion Backbone
	· Phase 1 anchor seeding success：展示条形图和热图。解释：成功率高/低的口袋对应后续 HEW 占用潜力。
	· HEW proximity and Pareto improvement：展示质心距离箱线图（图 Y）、COS 累积分布图（图 Z）、帕累托散点图（图 W）。表格：三个条件平均指标 + p 值。
	· Physical plausibility and drug‑likeness：展示 strain、clash、vina、QED、SA 的表格。强调 KAG 在 strain 和 vina 方差上的优势。
· 4.4 Results  on Pharmacophore-Constrained Generation：实验设置（口袋、药效团点定义、基线方法），展示结果图表（接近度、strain、QED、SA、帕累托分析...），与hard fix、全梯度对比。（和4.2以及4.3大致一致）

5. Related Work

· 结构基分子生成（Decompdiff, DrugFlow）、推理时引导（硬固定、Lai+SoftFix, BADGER, KPE）。


6. Discussion
· 重申贡献：两阶段运动学锚定、零应变保证、跨模型跨场景即插即用。
· 总结主要发现：KAG 实现了 strain 和 HEW 接近度的帕累托改进。
· 局限性：Phase 1 锚点成功率有限；单一位点引导无法同时覆盖多个 HEW。
· 未来方向：多片段连接、软内弛豫、学习兼容性矩阵。


附录 (Supplementary Information)

· 放置所有消融实验（A1‑A6 的表格和图）
· 额外的口袋结果（如 2gqn, 6phx 的帕累托图）
· Phase 1 每个位点覆盖的热图
· 硬固定实现对比的箱线图（若有）
· 敏感性分析（COS 对 σ）
· 推理时间对比
· 启发式矩阵

---

最后说明

请按照上述四个部分逐步执行。每完成一部分，请输出中间结果（例如代码、CSV 文件、图表）并确认。所有图表请保存为高分辨率 PNG 或 PDF，命名规则：fig_{章节}_{内容}.png。
