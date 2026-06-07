---
name: esfield-code-development
description: ESField 项目代码开发与实验执行规范。Use when Codex needs to implement, modify, review, test, run, or document code for the ESField energy-site-guided 3D molecular generation project，包括 Python 模块、配置文件、数据预处理、site map、atom-site pair、势能网络、flow matching guidance、评估脚本、服务器运行、smoke test、开发文档维护、中文阶段说明、git 与实验记录。
---

# ESField 代码开发规范

## 必读上下文

在修改代码、运行脚本或更新实验流程前，先阅读：

1. `开发者文档.md`：当前状态、路径策略、GPU 策略、交接规则。
2. `研究计划.md`：方法边界、MVP 成功标准、实验路线。
3. `任务看板.md`：当前优先级、待确认事项、下一步任务。
4. `references/esfield_contract.md`：实现数据、模型、guidance、评估模块时必须遵守的实现契约。

如果这些文件与用户最新要求冲突，以用户最新要求为准，并在完成修改后同步更新文档。

## 语言规范

- 默认用中文写所有面向用户和面向协作 AI 的说明，包括阶段总结、开发者文档、任务看板、实验运行记录、handoff、失败原因、验证结果、README 类说明和论文相关说明。
- 代码标识符保持英文，包括文件名、模块名、类名、函数名、变量名、配置键、CLI 参数、日志字段名和 schema 字段名。
- 代码注释和 docstring 优先简洁。解释业务意图、实验假设、危险操作或非显然逻辑时可以用中文；通用类型说明和 API 说明可以用英文。
- 终端命令、错误日志、配置键、包名、数据字段名保持原样，不翻译。
- 如果生成阶段性说明文件，必须使用中文，并明确写出：完成了什么、如何验证、哪些没做、下一步是什么、风险是什么。

## 工作循环

1. 先检查现有文件，再编辑。
2. 文件修改前，用中文简要说明本次要改什么和为什么。
3. 保持改动范围清晰，一次只完成一个连贯任务。
4. 每个新行为都要新增或更新 smoke test。
5. 先运行最小但有效的验证。
6. 更新 `任务看板.md`；如果改变工程规范、目录、运行方式或关键决策，也更新 `开发者文档.md`。
7. 如果当前目录是 git 仓库，开始前检查 `git status --short`；在用户要求实现代码或明确允许提交时，在干净里程碑处提交。

## 代码质量要求

- 模块要小。单个 Python 文件目标控制在 300-400 行以内；职责分叉时尽早拆分。
- 函数要聚焦。函数目标控制在 80 行以内；复杂流程拆成小函数。
- 项目 schema 使用 dataclass、pydantic model 或 typed dict，不要在内部 API 到处传散乱嵌套 dict。
- CLI 入口放在 `scripts/` 或模块 `main()`；可复用逻辑放在 `src/`。
- 所有路径从配置、CLI 参数或环境变量读取，不写死个人路径。
- 遍历 CrossDocked 或大型目录的脚本必须支持 `--limit`、`--dry-run`，并输出清晰进度。
- 错误信息必须可操作：说明缺哪个文件、期望什么格式、哪个配置键有问题、建议如何修复。
- 不虚构实验数值。没有真实运行就写“未运行”“待运行”或“运行失败”。

## ESField 方法边界

- 势能网络学习的是 atom-site compatibility，不是真实物理自由能。
- 位点检测是离线预处理；势能网络不负责发现位点。
- MVP 位点类型只包括 `high_energy_water`、`stable_water`、`hydrophobic_cavity`。
- MVP 的 guidance 只修改坐标，不直接修改 atom type logits。
- MVP 中 stable water 只做保护；显式 water-bridge attraction 是后续工作。
- 评估必须同时报告 site-specific metrics、分子合法性、几何质量、SA score，以及可用的 docking proxy。

## 数据与产物策略

- 大文件不进 git：数据集、生成分子、docking 输出、checkpoint、wandb 日志、临时缓存。
- 小规模本地运行可使用 `data/raw`、`data/processed`、`data/site_maps`、`data/atom_site_pairs`。
- 服务器上优先使用环境变量：`ESFIELD_DATA_DIR`、`ESFIELD_RUN_DIR`、`ESFIELD_CHECKPOINT_DIR`。
- 每次实验输出都保存配置快照。
- 未经用户明确要求，不删除 raw data 或 checkpoint。

## 验证策略

用能证明改动的最小验证：

- Schema 改动：构造 toy object，做 JSON 序列化、反序列化和等价检查。
- Parser 改动：用极小 fixture 或一个真实样例配合 `--limit 1` 运行。
- Pair builder：在 toy case 上断言正负样本数量、标签、距离阈值。
- Model 改动：用 synthetic tensor 跑一次 forward 和 backward。
- Guidance 改动：确认 `E_total` 有限、`grad_E` 形状正确、梯度范数被裁剪。
- Evaluation 改动：用 toy molecule/site 或预计算最小样例验证指标。

普通编码不需要 GPU。只有训练模型、生成器推断、guided sampling 或大规模评估时才使用 GPU。

## 长任务规范

任何预计超过 10 分钟的服务器任务：

1. 使用 `tmux`。
2. 在 `实验运行记录.md` 记录命令、配置、输出目录、开始时间、预计耗时和关机计划。
3. 启动后观察前几分钟的错误、GPU 显存、CPU 和磁盘。
4. 付费 GPU 实例必须在用户确认后设置关机计时器或云平台自动关机。
5. 任务结束后用中文总结日志、结果和失败原因，更新任务状态，再停机。

## 需求变更处理

当需求变化时：

1. 先在 `任务看板.md` 记录新决策，并标记被替代的旧任务。
2. 行为变化先改配置，再改代码。
3. 成本不高时，为已有中间文件保留向后兼容读取。
4. Schema 改动必须增加或保留 `schema_version`，并写迁移说明。
5. 不要用新规则静默重新解释旧结果。

## Git 里程碑

当项目处于 git 仓库中：

- 完成一个可运行模块和 smoke test 后再提交。
- 只 stage 本任务相关文件。
- 提交信息可以保留英文 conventional commit，例如 `feat: add site map schema`；提交正文或说明用中文。
- 不提交大型生成文件。
- 如果测试无法运行，要在最终回复和 `任务看板.md` 中用中文说明原因。
