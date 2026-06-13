# Alpha Mining / Factor Analyze 项目使用手册

本文档是当前项目的完整本地使用手册，面向“在单机环境中做 A 股因子研究、闭环挖掘、结果复核、Superalpha 组合与 Live shadow 演练”的使用者。

> 重要边界：本项目是本地量化研究工作台，不是自动交易系统。当前 Live 能力是 shadow/manual-review 工作流，不连接券商、不自动下单、不产生真实成交回报。

## 1. 项目定位

项目核心链路：

```text
Data Lake -> DuckDB -> Universe Store -> Closed Loop -> Analysis Run -> Factor Library -> Superalpha -> Live
```

各层含义：

- `Data Lake`：本地 Parquet 数据湖，保存 Tushare 原始表、标准化 curated 表、快照和 field catalog。
- `DuckDB`：本地查询层，主要文件为 `data/duckdb/market.duckdb`，用于构建研究面板视图。
- `Universe Store`：因子研究结果目录，默认在 `data/alpha_universe_store`。
- `Closed Loop`：自动生成、筛选、分析、反馈迭代因子的闭环流程。
- `Analysis Run`：一次因子分析产物，包含 metrics、PnL、meta、dashboard compact table 等。
- `Factor Library`：通过准入规则后的因子注册库，供 Superalpha 复用。
- `Superalpha`：由多个 accepted 因子组合成的研究组合。
- `Live`：对 active Superalpha 生成单日 signal、holdings、manual-review orders 的 shadow-production 层。

## 2. 安全边界与使用原则

- Dashboard 只建议绑定 `127.0.0.1`，不要暴露到局域网、VPN、反向代理或公网。
- 不要把 Tushare token、账户文件路径、私有 endpoint 写入提交文件。
- `configs/datasource.local.yaml` 是本地文件，项目不会自动修改它；secret 建议通过环境变量传入。
- Live shadow run 不自动提交订单；生成的 orders 仅供人工检查。
- 数据与 artifact 体量会持续增长，清理前先用只读审计脚本确认来源。

## 3. 目录结构速览

```text
alpha_mining/                    核心挖掘、数据源、workflow、Live 逻辑
factor_research/                 单因子分析、诊断、组合分析基础能力
dashboard/api/                   FastAPI 后端
dashboard/frontend/              React dashboard 前端
scripts/                         常用命令行入口
configs/                         示例配置和本地配置
docs/                            专题文档、runbook、maintenance checklist
tests/                           dashboard/API/文档/脚本测试
data/lake/                       Parquet 数据湖
data/duckdb/market.duckdb        DuckDB 查询层
data/alpha_universe_store/       universe、closed-loop、analysis、Live 结果
artifacts/                       质量检查、开发审计、临时报告
logs/                            手动运行留下的历史日志
```

## 4. 环境准备

### 4.1 Python 环境

项目默认使用本地虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

如需重建环境，通常流程是：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4.2 前端环境

Dashboard 前端位于 `dashboard/frontend`。

首次或依赖缺失时：

```powershell
cd dashboard\frontend
npm install
cd ..\..
```

构建前端：

```powershell
cd dashboard\frontend
npm run build
cd ..\..
```

### 4.3 PowerShell 中文显示

如果 PowerShell 读取中文文档出现乱码，先设置输出编码：

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 5. 配置与 Secret

### 5.1 数据源配置

复制模板：

```powershell
copy configs\datasource.example.yaml configs\datasource.local.yaml
```

推荐把 token 放到环境变量：

```powershell
$env:TUSHARE_TOKEN = "<your_token>"
```

然后运行 preflight：

```powershell
.\.venv\Scripts\python.exe scripts\preflight_guard.py --strict
```

预期结果：

- 无 warning：通过。
- 如果提示 `local datasource config contains non-empty tushare.token`：清空 `configs\datasource.local.yaml` 里的 `tushare.token`，改用 `$env:TUSHARE_TOKEN`。
- 如果提示 custom `tushare.http_url`：这是 info，确认 endpoint 是你预期的私有地址即可。

### 5.2 Live 配置

复制 Live 模板：

```powershell
copy configs\live.local.example.yaml configs\live.local.yaml
```

Live 配置主要控制：

- universe 与 store root
- DuckDB 路径与 source view
- active Superalpha 数量上限
- target holdings 数量、权重、现金 buffer
- tradability gate
- account/position 文件
- manual-review orders
- parity smoke check

## 6. 数据层工作流

### 6.1 全量构建数据湖

适合首次构建或重建数据：

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_tushare_lake.py `
  --config configs\datasource.local.yaml `
  --start-date 2016-01-01 `
  --end-date 2026-04-21 `
  --include-p2 `
  --flush-trade-days 20
```

行为：

- 拉取 P0/P1 基础表。
- 可选拉取 P2 财务表。
- 写入 `data/lake/vendor_raw` 和 `data/lake/curated`。
- 重建 DuckDB catalog。
- 默认支持 checkpoint resume。

### 6.2 增量刷新数据

日常刷新：

```powershell
.\.venv\Scripts\python.exe scripts\update_tushare_lake.py `
  --config configs\datasource.local.yaml `
  --refresh-dims `
  --flush-trade-days 20
```

只预览计划，不实际拉取：

```powershell
.\.venv\Scripts\python.exe scripts\update_tushare_lake.py `
  --config configs\datasource.local.yaml `
  --fact-groups p1 `
  --dim-groups p1 `
  --refresh-dims `
  --dry-run
```

### 6.3 重建 DuckDB catalog

```powershell
.\.venv\Scripts\python.exe scripts\build_duckdb_catalog.py --config configs\datasource.local.yaml
```

主要输出：

```text
data/duckdb/market.duckdb
```

### 6.4 刷新 Field Catalog

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_catalog.py --config configs\datasource.local.yaml
```

主要输出：

```text
data/lake/meta/field_catalog.parquet
```

### 6.5 刷新 Field Coverage

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_coverage.py --config configs\datasource.local.yaml
```

如果 Dashboard 的 Data Health 显示 `Not refreshed`，优先考虑刷新 field catalog/coverage 或运行质量检查。

### 6.6 Panel Quality 检查

```powershell
.\.venv\Scripts\python.exe scripts\check_panel_quality.py --config configs\datasource.local.yaml
```

质量检查结果通常位于：

```text
artifacts/data_quality/
```

## 7. 启动 Dashboard

### 7.1 构建前端

```powershell
cd dashboard\frontend
npm run build
cd ..\..
```

### 7.2 启动本地服务

```powershell
.\.venv\Scripts\python.exe scripts\run_factor_dashboard.py `
  --store-root data\alpha_universe_store `
  --host 127.0.0.1 `
  --port 8008
```

浏览器打开：

```text
http://127.0.0.1:8008
```

安全提醒：

- 只绑定 `127.0.0.1`。
- 不要把 dashboard 暴露到公网。
- Dashboard 能启动本地 closed-loop job、查看本地 artifacts、激活 Live Superalpha，因此不要当成公开 Web 服务。

## 8. Dashboard 页面说明

### 8.1 Overview

Overview 用于快速判断工作台是否可用：

- universe 数量
- run 数量
- 最新分析 run 时间
- field catalog 新鲜度
- Live latest 状态
- preflight 状态
- freshness warnings

常见状态：

- `preflight warning`：先修 secret/config。
- `field catalog stale`：刷新 field catalog。
- `live_missing`：当前没有 active/latest Live artifact。
- `no_analysis_runs`：当前 universe 没有可看的分析结果。

### 8.2 Alphas

Alphas 用于浏览因子结果：

- 选择 universe。
- 选择 run 或 scoreboard。
- 搜索因子。
- 查看 score、tier、IC、Sharpe、turnover、coverage 等指标。
- 打开因子详情抽屉查看 PnL、analysis data、visual artifacts。

有效因子通常由：

- tier 为 `S/A/B`
- 或 `score >= 60`

作为 dashboard 的有效筛选口径。

### 8.3 Closed Loop

Closed Loop 用于从 dashboard 启动后台挖掘任务。

建议首次使用 preset：

- `Smoke`：小规模冒烟，适合确认环境。
- `Balanced`：日常本地试跑。
- `Deep`：更重的探索，运行前确认内存、磁盘、DuckDB temp。

关键参数：

- `universe`：研究 universe。
- `request_new`：请求生成的新因子数。
- `batch_size`：每批处理数量，应小于等于 `request_new`。
- `max_eval`：候选评估数量，越大越慢、越占空间。
- `iterations`：闭环迭代次数。
- `source_chunk_loading`：建议启用。
- `source_chunk_mem_hard_limit_mb`：dashboard 本地安全默认建议非零，例如 `4096`。

Job 目录：

```text
data/alpha_universe_store/_dashboard_jobs/closed_loop/<job_id>/
```

常见文件：

```text
job.json
request.json
command.json
stdout.log
stderr.log
```

常见状态：

- `queued`：已创建，等待启动。
- `running`：正在运行。
- `succeeded`：成功完成。
- `failed`：失败，优先看 failure hint 和 stderr tail。
- `cancelled`：用户取消。
- `interrupted`：dashboard 恢复到 running 记录，但进程已不在。
- `Running outside dashboard`：进程仍可能存活，但不在当前 FastAPI 内存进程表中。

失败排查顺序：

1. 看 `status_label/status_hint`。
2. 看 `failure_category/failure_hint`。
3. 看 `stderr.log` tail。
4. 看 lock owner。
5. 看 `run_health.jsonl`。
6. 再决定降低 `max_eval/batch_size`，或调整 DuckDB temp/memory。

### 8.4 Compare

Compare 用于比较同一 universe 下两个真实 analysis run。

限制：

- 只比较真实 analysis run。
- 不比较 `__scoreboard__`。

主要内容：

- run summary
- metric delta
- top factor overlap

Artifact status：

- `Complete`：可正常比较。
- `Partial metrics`：部分指标缺失，缺失值不要理解成表现差。
- `Missing artifact`：关键 artifact 缺失，先检查产生该 run 的流程。
- `Invalid artifact`：artifact 存在但不可用于比较。

### 8.5 Data

Data 页面包含：

- `Catalog`：字段浏览、搜索、family/role/coverage 信息。
- `Health`：只读健康摘要。

Health 关注：

- catalog row count
- searchable count
- average coverage
- coverage missing/partial/available
- family health
- base frame 文件是否存在
- parquet rows/columns
- run_health 最近摘要
- data quality artifact

状态解释：

- `Not refreshed`：coverage/freshness metadata 尚未刷新，不等于源数据一定错误。
- `base frame missing`：当前 universe 缺少 base frame。
- `run_health missing`：该 universe 没有 closed-loop health artifact。
- `quality artifact warning`：最新质量检查存在 warning/fail。

### 8.6 Research

Research 用于查看更偏研究/诊断的输出。具体可用内容依赖已有 analysis artifacts。

### 8.7 Superalpha

Superalpha 用于从 accepted factors 组合研究组合：

- 选择 accepted factors。
- 设置组合权重表达式。
- 运行 backtest。
- 查看 PnL、metrics、analysis data。
- 将合适的 Superalpha 激活到 Live registry。

注意：

- 因子缺少 compact signal 时可能需要 reproduce fallback。
- Reproduce fallback 可能从 DuckDB 加载较多数据，内存占用更高。
- 单因子 Superalpha 可用于 smoke test，多因子组合更适合研究。

### 8.8 Live

Live 页面用于查看 shadow-production 状态：

- active Superalphas
- data readiness
- latest signals
- latest holdings
- orders review status
- stale jobs
- field catalog warnings

Live 不提交真实订单，只生成人工复核材料。

## 9. 命令行 Closed Loop

Dashboard 是推荐入口，但 CLI 仍兼容。

最小本地 DuckDB 例子：

```powershell
.\.venv\Scripts\python.exe scripts\run_closed_loop.py `
  --source-backend duckdb `
  --datasource-config configs\datasource.local.yaml `
  --duckdb-path data\duckdb\market.duckdb `
  --source-view v_project_panel_cn_a `
  --start-date 2022-01-01 `
  --end-date 2026-04-21 `
  --universe cn_all `
  --request-new 5 `
  --batch-size 5 `
  --max-eval 80 `
  --iterations 1 `
  --source-chunk-loading `
  --source-chunk-mem-hard-limit-mb 4096
```

推荐原则：

- 第一次先用小参数。
- `batch_size <= request_new`。
- 优先启用 `source_chunk_loading`。
- 本地机器内存有限时，降低 `max_eval` 和 `request_new`。
- 不确定时先通过 dashboard preset 跑 Smoke。

## 10. Factor Library 准入规则

Factor Library 默认关闭，closed-loop CLI 可通过相关参数启用。

当前准入语义：

- `accepted`：`score >= min_score` 且相关性阈值通过，默认 `min_score = 60`。
- `staging`：score 在 50-60，或 corr 在 staging 区间。
- `rejected`：不满足 staging。
- 历史低于当前阈值但状态为 accepted 的记录，应理解为 `legacy accepted`。

默认相关性要求：

- `signal_corr < 0.80`
- `ic_corr < 0.80`
- `max_pnl_corr < 0.80`

当交易成本启用且 net metrics 可用时，准入优先使用 net score。

## 11. Superalpha 与 Live Shadow 工作流

### 11.1 检查 Live readiness

```powershell
.\.venv\Scripts\python.exe scripts\check_live_readiness.py `
  --config configs\live.local.yaml `
  --universe cn_all
```

### 11.2 运行 Live Superalpha dry-run

```powershell
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py `
  --config configs\live.local.yaml `
  --universe cn_all `
  --superalpha-id superalpha_xxx `
  --dry-run `
  --skip-parity
```

常见输出：

- live signal
- target holdings
- manual-review orders
- latest pointer
- job state

### 11.3 检查 parity

```powershell
.\.venv\Scripts\python.exe scripts\check_live_superalpha_parity.py `
  --config configs\live.local.yaml `
  --universe cn_all `
  --superalpha-id superalpha_xxx
```

### 11.4 人工验收

阅读：

```text
docs/live_superalpha_runbook.md
```

人工检查重点：

- active Superalpha 是否正确。
- resolved signal date / common ready date 是否符合预期。
- holdings 数量、权重、现金 buffer 是否合理。
- orders 是否只用于人工复核。
- account/position 文件是否新鲜。
- Live 与回测信号 parity 是否可接受。

## 12. 结果与 Artifact 位置

### 12.1 Universe Store

```text
data/alpha_universe_store/<universe>/
```

常见子目录：

```text
catalog/
analysis/
feedback/
library/
superalphas/
live/
base/
```

### 12.2 因子 registry

```text
data/alpha_universe_store/<universe>/catalog/expressions.csv
```

### 12.3 候选 artifacts

```text
data/alpha_universe_store/<universe>/catalog/candidates/
```

### 12.4 Analysis artifacts

```text
data/alpha_universe_store/<universe>/analysis/period_<N>/analysis_*/
```

常见文件：

```text
analysis_meta.json
factor_metrics.csv
dashboard_factor_metrics.csv
portfolio_pnl_df.parquet
phase_metrics_df.parquet
```

### 12.5 Feedback scoreboard

```text
data/alpha_universe_store/<universe>/feedback/expression_scoreboard.csv
```

### 12.6 Run health

```text
data/alpha_universe_store/<universe>/feedback/run_health.jsonl
```

### 12.7 Live artifacts

```text
data/alpha_universe_store/<universe>/live/
```

常见路径：

```text
active_superalphas.json
signals/<superalpha_id>/
holdings/<superalpha_id>/
orders/<superalpha_id>/
jobs/<superalpha_id>/
latest.json
```

## 13. 数据与空间治理

主要增长来源：

- `data/lake`：原始表、curated 表、snapshots、field catalog。
- `data/duckdb`：DuckDB database 和 temp/spill 目录。
- `data/alpha_universe_store`：closed-loop candidates、analysis runs、factor library、Superalpha、Live shadow artifacts。
- `dashboard job logs`：dashboard-launched closed-loop job 的 stdout/stderr/job meta。

只读审计 analysis artifact：

```powershell
.\.venv\Scripts\python.exe scripts\audit_analysis_artifacts.py --store-root data\alpha_universe_store
```

检查 closed-loop health：

```powershell
.\.venv\Scripts\python.exe scripts\inspect_closed_loop_health.py --store-root data\alpha_universe_store --universe cn_all
```

压缩 Parquet lake：

```powershell
.\.venv\Scripts\python.exe scripts\compact_parquet_lake.py --config configs\datasource.local.yaml
```

清理建议：

- 优先清理 `__pycache__`、`.pytest_cache`、frontend `dist`、临时 log。
- 不要随手删除 `data/lake`、`data/duckdb/market.duckdb`、`data/alpha_universe_store`。
- 删除某个 universe 前，确认其中是否有 active Superalpha、factor library、Live artifacts。
- 对 analysis artifacts，优先使用已有 retention 配置或审计脚本判断。

## 14. 常见状态与处理

| 状态 | 含义 | 建议 |
| --- | --- | --- |
| `preflight warning` | secret/config 有风险 | 清空 local YAML token，改用环境变量 |
| `coverage not refreshed` | coverage metadata 缺失或过期 | 刷新 field catalog/coverage 或运行质量检查 |
| `Missing artifact` | 预期文件不存在或不可读 | 回到产生该 run/job 的流程检查 |
| `Partial metrics` | 部分指标列缺失 | 不要把缺失值解释成表现差 |
| `memory hard limit` | chunk memory 保护触发 | 降低 `max_eval/request_new/batch_size` |
| `DuckDB temp` | DuckDB spill/temp 受限 | 设置 temp dir 或增大 temp limit |
| `stale lock` | lock heartbeat 老旧或进程不明 | 先检查 lock owner，不要直接误删 |
| `Interrupted` | dashboard 记录 running，但进程已不在 | 看 stderr tail 和 run artifacts 后重跑 |
| `Running outside dashboard` | 进程仍可能存活，但不在当前 API 内存表 | 确认 PID 状态，不要重复启动同 universe 重任务 |
| `Live missing` | 无 active/latest Live artifact | 激活 Superalpha 或运行 Live shadow |

## 15. 常用脚本索引

| 脚本 | 用途 |
| --- | --- |
| `scripts/preflight_guard.py` | 检查 secret/config 风险 |
| `scripts/bootstrap_tushare_lake.py` | 首次构建数据湖 |
| `scripts/update_tushare_lake.py` | 增量刷新 Tushare lake |
| `scripts/build_duckdb_catalog.py` | 构建 DuckDB catalog |
| `scripts/refresh_field_catalog.py` | 刷新字段 catalog |
| `scripts/refresh_field_coverage.py` | 刷新字段 coverage |
| `scripts/check_panel_quality.py` | 检查 panel 质量 |
| `scripts/run_factor_dashboard.py` | 启动 dashboard |
| `scripts/run_closed_loop.py` | CLI 闭环挖掘 |
| `scripts/audit_analysis_artifacts.py` | 只读审计 analysis artifacts |
| `scripts/inspect_closed_loop_health.py` | 查看 closed-loop health |
| `scripts/reproduce_alpha.py` | 复现单因子 |
| `scripts/reproduce_alpha_oos.py` | OOS 复现 |
| `scripts/check_operator_registry.py` | 检查 operator/signature registry |
| `scripts/audit_wq_operator_coverage.py` | 审计 WQ operator 覆盖 |
| `scripts/run_live_superalpha.py` | 运行 Live shadow |
| `scripts/check_live_readiness.py` | 检查 Live readiness |
| `scripts/check_live_superalpha_parity.py` | 检查 Live parity |

## 16. 测试与验证

### 16.1 Dashboard API 回归

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_dashboard_next_workbench_api.py `
  tests\test_dashboard_workbench_api.py `
  tests\test_factor_dashboard_api.py `
  tests\test_dashboard_live_api.py `
  tests\test_superalpha_dashboard_api.py `
  -q
```

### 16.2 核心包测试

```powershell
.\.venv\Scripts\python.exe -m pytest alpha_mining\tests -q
```

### 16.3 全量测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### 16.4 前端构建

```powershell
cd dashboard\frontend
npm run build
cd ..\..
```

## 17. 推荐日常操作节奏

### 每次打开项目

1. 启动 dashboard。
2. 看 Overview。
3. 看 preflight。
4. 看 field catalog freshness。
5. 看是否有 running/failed closed-loop job。

### 每次跑闭环前

1. 确认数据新鲜度。
2. 确认 Data Health 没有关键 missing。
3. 首次先用 Smoke/Balanced preset。
4. 保持 chunk loading 开启。
5. 给 hard limit 设置非零值。

### 每次解释结果前

1. 看 run artifact status。
2. 看 score basis 是 gross 还是 net。
3. 看 train/validation/OOS。
4. 看 transaction cost 后表现。
5. 看 factor library status。
6. 对 Superalpha 结果，额外看 component signal status 和 reproduce warning。

### 每周维护

1. 跑 artifact audit。
2. 看 `data/lake`、`data/duckdb`、`data/alpha_universe_store` 体量。
3. 检查 run_health 是否连续出现 hard limit。
4. 检查 Live latest 是否过期。
5. 跑一轮 dashboard API 回归或全量测试。

## 18. 当前能力边界

已具备：

- A 股日频研究链路。
- Tushare Parquet lake。
- DuckDB panel view。
- Dashboard 本地工作台。
- Closed-loop 因子挖掘。
- 因子库准入。
- Run Compare。
- Data Health。
- Superalpha backtest。
- Live shadow/manual-review artifacts。

未完成或不应误解为已完成：

- 不支持自动券商下单。
- 不支持真实 live PnL/fills。
- 不支持分钟级或多市场生产级链路。
- Field metadata 是规则推断，不是完整 point-in-time 数据血缘证明。
- 因子评分是研究工程口径，不等于投资建议。

## 19. 推荐阅读顺序

首次使用：

1. `README.md`
2. `docs/quickstart.md`
3. 本手册
4. `docs/maintenance.md`

做因子研究：

1. `docs/field_catalog.md`
2. `docs/operator_registry_audit.md`
3. `docs/experiment_artifacts_schema.md`

做 Live shadow：

1. `docs/live_superalpha_runbook.md`
2. `configs/live.local.example.yaml`

维护和排障：

1. `docs/maintenance.md`
2. Dashboard Overview / Data Health / Closed Loop detail
3. `scripts/audit_analysis_artifacts.py`
4. `scripts/inspect_closed_loop_health.py`
