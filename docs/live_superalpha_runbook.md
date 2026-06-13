# Live SuperAlpha 运行手册

Live SuperAlpha 是影子/生产准备工作流。它不连接券商 API、不提交订单、不更新 Tushare 数据。

## 每日操作流程

1. 使用现有数据源工作流更新数据湖和 DuckDB 目录。

```powershell
.\.venv\Scripts\python.exe scripts\update_tushare_lake.py --config configs\datasource.local.yaml --refresh-dims --flush-trade-days 20
```

2. 运行只读就绪检查。

```powershell
.\.venv\Scripts\python.exe scripts\check_live_readiness.py --config configs\live.local.yaml --universe cn_all --position-path data\alpha_universe_store\cn_all\live\account\positions_current.csv --account-total-value 1000000 --cash 200000
```

3. 运行 Live 生产准备任务。

```powershell
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py --config configs\live.local.yaml --universe cn_all --date auto
```

Live 脚本仅读取 live 配置和现有 alpha universe store。它不读取 `configs/datasource.local.yaml`，不使用 Tushare 凭据。

信号对称性冒烟检查：将窗口化的 live 截面与参考 SuperAlpha 信号比较：

```powershell
.\.venv\Scripts\python.exe scripts\check_live_superalpha_parity.py --config configs\live.local.yaml --universe cn_all --superalpha-id superalpha_xxx --date 2026-05-25
```

订单仅为手动审核产物。它们需要显式的 account/position 基础；否则系统发出权重差值而无可靠的订单价值或股数。

```powershell
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py --config configs\live.local.yaml --universe cn_all --superalpha-id superalpha_xxx --position-path data\alpha_universe_store\cn_all\live\account\positions_current.csv --cash 200000 --account-total-value 1000000
```

当多个 SuperAlpha 激活时，订单默认为单个 `--superalpha-id` 范围，除非账户和分配被显式分离。这防止意外跨策略复用同一账户快照。

使用 `configs/live.local.example.yaml` 作为本地模板，保存为 `configs/live.local.yaml`。不要提交真实的账户值或持仓导出。持仓文件必须遵循 `docs/live_positions_template.csv`，至少包含 `code`、`shares`、`available_shares`、`last_price` 或 `market_value`，以及 `position_date` 或 `updated_at`。

## 每日命令模式

```powershell
# 仅持仓 dry-run
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py --config configs\live.local.yaml --universe cn_all --superalpha-id superalpha_xxx --dry-run --skip-orders --skip-parity

# 订单 dry-run（手动审核）
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py --config configs\live.local.yaml --universe cn_all --superalpha-id superalpha_xxx --dry-run --skip-parity --position-path data\alpha_universe_store\cn_all\live\account\positions_current.csv --account-total-value 1000000 --cash 200000
```

Dashboard Live 应用于检查数据状态、选定市值字段、字段目录警告、最新持仓、订单可审核性、阻止原因和任务过期/错误状态。

## 人工验收

- 对于新的或最近更改的 SuperAlpha，在发布正式每日产物之前至少运行一次对称性冒烟测试。
- 对于订单，在执行任何本项目外的手动操作之前，验证账户总值、现金、持仓日期、阻止买入/卖出数量、预估费用和现金余额。
- 如果就绪检查返回 `BLOCKED`，除非阻止字段/原因已被理解并使用 `--force-reason` 记录，否则不要强制运行。

产物写入路径：

```text
data/alpha_universe_store/{universe}/live/
```

每个活跃的 SuperAlpha 有独立的最新指针用于信号、持仓、订单和任务状态。失败的任务不会覆盖之前成功的持仓或订单最新值。dry-run 不发布官方最新指针。
