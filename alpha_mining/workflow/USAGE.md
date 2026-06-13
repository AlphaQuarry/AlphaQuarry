# Alpha Mining Workflow 使用手册

本项目是本地因子挖掘与分析工具。官方工作流路径为 Python workflow + CLI：

`closed-loop mining -> analysis_cycle.py -> analysis run artifacts -> FastAPI dashboard -> React dashboard`

Notebook 是研究用的临时工具，不是正式本地运行的权威数据源。

## 官方入口

正式运行请使用以下入口：

- `alpha_mining.workflow.run_factor_analysis_batch`
- `alpha_mining.workflow.run_one_loop_iteration`
- `alpha_mining.workflow.run_closed_loop`
- `scripts/run_closed_loop.py`

Dashboard 期望分析产物写入以下路径：

`data/alpha_universe_store/<universe>/analysis/period_<period>/<analysis_run_id>/`

基于文件的 CLI 运行示例：

```powershell
cd D:\project_quant
.venv\Scripts\python.exe scripts\run_closed_loop.py `
  --source-backend file `
  --data-path path\to\panel.parquet `
  --universe cn_all `
  --base-dir data\alpha_universe_store `
  --iterations 1 `
  --request-new 5 `
  --batch-size 5
```

基于 DuckDB 的运行请继续使用 `scripts/run_closed_loop.py` 已支持的数据源参数，如 `--duckdb-path`、`--source-view`、`--start-date`、`--end-date` 和 `--run-filters-json`。

## 训练 / 验证 / 测试

默认的阶段窗口是固定的：

- `train`：`2016-01-01` 至 `2024-12-31`
- `val`：`2025-01-01` 至 `2025-12-31`
- `test`：`2026-01-01` 至运行的最后一个回测日期

`val` 仅在运行达到 `2025-01-01` 时出现。`test` 仅在运行达到 `2026-01-01` 时出现。

阶段指标默认启用。禁用方式：

```powershell
.venv\Scripts\python.exe scripts\run_closed_loop.py --no-phase-metrics
```

闭环反馈范围默认为训练期：

```text
当前有效的 score_total / feedback_score 别名
```

启用交易成本时，当前有效别名优先使用 `score_total_net` / `feedback_score_net`。
未启用交易成本或缺少 net 字段时，回退到兼容的 gross/legacy `score_total` 路径。
验证期和测试期指标仅用于诊断，不能选择默认反馈目标。

正式方向选择为 `train_locked`：训练期决定用于评分、反馈、排序和 Dashboard 默认值的方向。`phase_local` 诊断可能显示每个阶段的本地方向/层，但这些诊断不进入正式的样本外评分。

交易约束默认启用。使用 `--no-tradability-constraints` 恢复旧的无限制行为。这可能导致与旧运行结果不同，但更符合 A 股的买卖约束。

基准比较从 `universe` 绑定（如果存在映射）。使用 `--benchmark-code 000300.SH` 显式恢复旧的沪深 300 比较。如果没有 universe 映射，工作流回退到沪深 300 并发出警告。

收益语义在 `analysis_meta.json` 中明确：本项目目前假设收盘后形成因子；`delay=1 + pct_chg` 评估延迟暴露后的下一个日收益。在转向盘中、日内或生产执行之前，需要收紧 `available_at` 和执行价格假设。

## 分析产物

每次正式分析运行保存全期指标和紧凑的阶段感知产物。

核心 Dashboard 产物：

- `factor_metrics.csv`
- `dashboard_factor_metrics.csv`
- `phase_metrics_df.csv`
- `ic_df.csv`
- `portfolio_pnl_df.parquet`
- `analysis_distribution_histogram.csv`
- `analysis_ic_decay.csv`
- `analysis_meta.json`

动态 Dashboard 读取：

- IC 概览来自 `ic_df.csv`
- 年度 IC 来自后端对 `ic_df.csv` 的聚合
- 分布来自 `analysis_distribution_histogram.csv`
- IC 衰减来自 `analysis_ic_decay.csv`
- 层终端收益来自 `portfolio_pnl_df.parquet`
- 阶段指标摘要来自 `phase_metrics_df.csv`

紧凑分布产物仅存储 bin/count 摘要，不存储原始因子值。这在保持低磁盘占用的同时，仍允许在 Dashboard 中进行 train/val/test 交互。

## 评分与成本基础

工作流在可用时保留 gross 和 net 评分列：

- `score_total_gross` / `feedback_score_gross`：gross 诊断评分
- `score_total_net` / `feedback_score_net`：扣费后评分
- `score_total` / `feedback_score`：当前默认有效基础的兼容别名

启用交易成本时，默认排序、反馈、阶段评分别名和因子库准入优先使用 net。Gross 仍可用于诊断和兼容性。只有 `score_total` / `feedback_score` 的旧产物仍可被 Dashboard 读取。

## 字段目录元数据

Data 页面读取本地字段目录而不重新计算覆盖率。`field_role`、`available_at`、`preprocessing_policy` 和 `leakage_safe` 目前是基于最小规则推断的提示，不是完整的数据血缘或公告时间验证系统。正式交易使用应连接真实公告时间戳和更严格的字段可用性检查。

## 因子库

因子库默认关闭，除非启用否则不写入接受的资产：

```powershell
.venv\Scripts\python.exe scripts\run_closed_loop.py --factor-library
```

接受的资产使用正式默认值：

- `min_score = 60`
- `signal_corr < 0.80`
- `ic_corr < 0.80`
- `max_pnl_corr < 0.80`，使用纯多头和多空诊断 PnL 相关性

阈值可通过 CLI/配置覆盖。评分为 50-60 或落入 0.80-0.95 相关性带的因子被记录为 `staging` 或 `rejected`，不被接纳到正式的 `accepted` 集合。启用交易成本时，准入评分选择优先使用 train-locked net 字段，如 `feedback_score_net` / `score_total_net`。

## SuperAlpha

SuperAlpha 允许将多个接受的因子组合成复合信号进行回测。

### 信号解析 (schema_version=2)

系统使用多级回退来解析组件信号：

1. **紧凑信号**（因子库注册表中的 `signal_artifact_path`）
2. **原始 alpha**（`alphas/{factor}.parquet`）
3. **组件缓存**（`superalphas/_component_cache/{factor}.parquet`）
4. **重生成**（使用 `reproduce_alpha_by_name` 从表达式重新生成）
5. **DuckDB 回退**（从 DuckDB 面板加载）

信号状态在 Dashboard 中显示为徽章：Compact / Raw / Cached / Reproducible / DuckDB Fallback / Unavailable。

### 配置

`SuperalphaConfig` 控制标准化和连接行为：

- `component_normalization`：组件标准化（`cs_zscore`，默认）
- `final_normalization`：最终信号标准化（`cs_zscore`，默认）
- `component_join`：连接方式（`inner`，默认）
- `direction_adjustment`：应用注册表/指标的方向符号（默认 `True`）
- `allow_reproduce_fallback`：允许通过 reproduce 重新生成信号（默认 `True`）
- `cache_reproduced_components`：缓存重生成的信号到 `_component_cache`（默认 `True`）
- `schema_version`：缓存键分离的哈希输入（默认 `2`）

### 方向解析

方向符号按优先级顺序解析：

1. 注册表 `direction_sign` 字段
2. 源运行指标 / 方向策略
3. 默认 `+1`（记录为 `direction_status=missing_default_positive`）

### 元数据权重

使用基于元数据的权重时（如 `score`、`feedback_score`）：

- 负值被截断为 0
- 全零权重会报错

固定权重（如 `[0.5, 0.3, 0.2]`）保留负值并按 `sum(abs(weight))` 标准化。

## PNG 可视化产物

静态 PNG 因子分析图像默认禁用。

显式生成 legacy PNG 产物：

```powershell
.venv\Scripts\python.exe scripts\run_closed_loop.py --include-visualization-png
```

启用时，运行还会写入：

- `visualizations/<factor>/*.png`
- `visualization_manifest.csv`

React Dashboard 优先使用动态 Analysis Data。PNG 文件仅是旧运行或显式启用运行的兼容回退。

## Dashboard 验证

UI 变更后构建前端：

```powershell
cd D:\project_quant\dashboard\frontend
npm run build
```

仅在需要手动浏览器检查时启动 Dashboard API：

```powershell
cd D:\project_quant
.venv\Scripts\python.exe -m uvicorn dashboard.api.app:app --host 127.0.0.1 --port 8010
```

然后打开：

`http://127.0.0.1:8010`

手动检查：

- 选择预期的 universe 和 analysis run
- 打开一个 factor drawer
- `PnL` 应显示 train/val 阶段背景
- 如果存在测试数据，`Show test period` 出现且默认关闭
- 隐藏测试时，PnL、Metrics 和 Analysis Data 不应显示测试数据
- 启用测试时，PnL、Metrics 和 Analysis Data 应包含测试
- `Analysis Data` 应在回退到 PNG 图像之前渲染动态图表

对于非服务器 API 验证，优先使用 FastAPI `TestClient` 或项目 pytest 测试，而不是长时间运行的前台 `uvicorn` 进程。

## 推荐检查

工作流或 Dashboard 变更后使用聚焦回归集：

```powershell
cd D:\project_quant
.venv\Scripts\python.exe -m pytest alpha_mining\tests\test_run_closed_loop_cli_args.py alpha_mining\tests\test_factor_library.py alpha_mining\tests\test_direction_policy.py alpha_mining\tests\test_analysis_score_basis.py alpha_mining\tests\test_field_catalog_builder.py alpha_mining\tests\test_analysis_data_artifacts.py tests\test_factor_dashboard_api.py alpha_mining\tests\test_workflow_analysis_cycle.py alpha_mining\tests\test_workflow_closed_loop.py::TestWorkflowClosedLoop::test_run_one_iteration_chunked_and_purged alpha_mining\tests\test_workflow_closed_loop.py::TestWorkflowClosedLoop::test_run_one_iteration_writes_visualization_png_when_enabled -q
```

预期的非阻塞警告：

- 来自旧 AST 兼容代码的 Parser 弃用警告
- 如果 `.pytest_cache` 不可写，Pytest 缓存警告

审计现有分析运行的产物磁盘使用：

```powershell
cd D:\project_quant
.venv\Scripts\python.exe scripts\audit_analysis_artifacts.py --store-root data\alpha_universe_store
```

有用的过滤器和机器可读输出：

```powershell
.venv\Scripts\python.exe scripts\audit_analysis_artifacts.py --store-root data\alpha_universe_store --universe cn_all --json
.venv\Scripts\python.exe scripts\audit_analysis_artifacts.py --store-root data\alpha_universe_store --csv-out artifacts\analysis_artifact_audit.csv
```

审计区分配置的 PNG 生成和实际的 PNG 文件，这对于在 PNG 生成默认禁用之前创建的旧运行很有用。

## 快速开始 (配置模板)

项目提供 3 个场景化配置模板，位于 `configs/` 目录：

| 模板 | 文件 | 适用场景 | 预计时间 |
|------|------|---------|---------|
| 快速验证 | `closed_loop_quick.yaml` | 首次使用、调试配置 | 2-5 分钟 |
| 深度挖掘 | `closed_loop_deep.yaml` | 正式研究、因子库建设 | 30-60 分钟 |
| 生产运行 | `closed_loop_production.yaml` | 每日自动运行 | 持续运行 |

使用方式：

```powershell
# 快速验证
.venv\Scripts\python.exe scripts\run_closed_loop.py --config configs\closed_loop_quick.yaml

# 深度挖掘 (启用闭环优化)
.venv\Scripts\python.exe scripts\run_closed_loop.py --config configs\closed_loop_deep.yaml

# 生产运行 (无限循环)
.venv\Scripts\python.exe scripts\run_closed_loop.py --config configs\closed_loop_production.yaml
```

## 闭环运行原理

### 确定性生成 + 反馈筛选

闭环运行的核心机制：

1. **候选生成**：`LayeredExpressionBuilder` 按 L0-L4 层级生成候选表达式
2. **预筛选**：语法检查、类型检查、去重
3. **评估**：对通过预筛选的候选进行因子分析
4. **反馈**：高分因子的特征 (字段、算子、子结构) 进入反馈权重
5. **下一轮**：反馈权重指导候选排序和字段选择

### 搜索空间优化

默认配置下，每轮生成约 532 个确定性候选，后续轮次重复。以下参数可优化搜索空间：

- `enable_feedback_mutation`：从高分因子片段生成新候选 (默认关闭)
- `field_rotation_focus_count`：每轮聚焦 N 个字段做完整探索 (默认 0 = 不轮转)
- `budget_rotation_mode`：不同轮次侧重不同层 (默认 none)

建议深度挖掘和生产运行启用这些优化：

```yaml
enable_feedback_mutation: true
field_rotation_focus_count: 3   # 建议 3-5
budget_rotation_mode: round_robin
```

## 长期运行注意事项

1. **搜索空间耗尽**：确定性候选在约 107 轮后耗尽 (532 候选 / 每轮评估 5 个)。启用 mutation 和轮转可大幅延长。
2. **因子衰减**：历史高分因子可能在未来失效。反馈机制的 `lookback_batches` 控制回看窗口。
3. **磁盘管理**：长期运行会积累大量分析产物。配置 `candidate_artifact_retention_days` 自动清理。
4. **fragment_registry 增长**：mutation 依赖的片段注册表会随时间增长。`mutation_fragment_max_age_batches` 控制最大年龄。

## 后续建议

1. 在真实的小型 universe 上运行审计，确认 PNG 默认禁用后的磁盘使用。
2. 为旧的 smoke 运行和 legacy PNG 密集运行添加可选的清理辅助工具。
3. 改进 Analysis Data 标签页的工具提示和空状态显示，不增加新的后端复杂性。
