# 快速上手

本指南面向本地单机工作流。保留现有命令行用法，同时提供 Dashboard 作为更安全的控制面板。

## 1. 启动 Dashboard

构建前端（首次）：

```powershell
cd dashboard\frontend
npm run build
cd ..\..
```

启动本地 Dashboard：

```powershell
.\.venv\Scripts\python.exe scripts\run_factor_dashboard.py --store-root data\alpha_universe_store --host 127.0.0.1 --port 8008
```

打开 `http://127.0.0.1:8008`，从 **Overview** 开始。Overview 显示 universe/run 数量、字段目录新鲜度、Live 可用性和预检警告。

保持 Dashboard 绑定到 `127.0.0.1`。它有本地写操作（如启动闭环任务），因此不要暴露到 LAN、VPN、反向代理或公共互联网。

日常使用的主要标签页：

- **Closed Loop**：启动和检查后台挖掘任务
- **Compare**：按摘要指标和 top 因子重叠比较两个分析运行
- **Data -> Health**：检查目录覆盖率、基础帧元数据、最近运行健康度和最新质量产物

日常/每周运维检查请参阅 [docs/maintenance.md](maintenance.md)。

## 2. 通过预检

运行：

```powershell
.\.venv\Scripts\python.exe scripts\reflight_guard.py --strict
```

预期：无警告。如果本地数据源配置包含 Tushare token，请清除 `configs\datasource.local.yaml` 中的 `tushare.token`，改为在 shell 中设置：

```powershell
$env:TUSHARE_TOKEN = "<your_token>"
```

Dashboard **Overview** 页面提供相同的非破坏性预检，但不显示密钥值。

## 3. 从 Dashboard 运行小型闭环

打开 **Closed Loop**，使用安全默认值：

- `source_backend`：`duckdb`
- `source_chunk_loading`：启用
- `source_chunk_mem_hard_limit_mb`：非零
- `request_new`、`batch_size`、`max_eval`、`iterations`：首次运行保持较小值

Dashboard 启动后台 Python 进程并将任务状态写入：

```text
data/alpha_universe_store/_dashboard_jobs/closed_loop/
```

FastAPI 进程仅调度和监控任务；挖掘进程负责大量内存使用。

任务失败时，请先查看 **Current State**、**Diagnosis**、**Lock Owner** 和 stderr 尾部，再修改参数。常见的 Dashboard 诊断包括内存保护、DuckDB 临时存储、数据过滤、锁冲突、配置错误、候选生成错误和分析产物错误。

如果任务显示 **Running outside dashboard**，表示 Dashboard 在重启后恢复了一个仍在运行的进程。如果显示 **Interrupted**，表示 Dashboard 恢复了一个运行中的任务记录，但进程已不存在。

## 4. 比较运行

在选定 universe 中至少有两个分析运行后，打开 **Compare**。页面会尽可能自动选择有指标产物和非零因子数量的近期运行。

产物状态很重要：

- **Complete**：比较指标可用
- **Partial metrics**：部分列缺失；缺失值显示为缺失产物，而非表现不佳
- **Missing artifact** 或 **Invalid artifact**：重新生成或检查分析运行后再解读差异

## 5. 刷新数据

当 Overview 页面报告数据过期时，运行数据湖更新并重建/刷新元数据：

```powershell
.\.venv\Scripts\python.exe scripts\update_tushare_lake.py --config configs\datasource.local.yaml --refresh-dims --flush-trade-days 20
.\.venv\Scripts\python.exe scripts\build_duckdb_catalog.py --config configs\datasource.local.yaml
.\.venv\Scripts\python.exe scripts\refresh_field_catalog.py --config configs\datasource.local.yaml
```

返回 **Overview** 并刷新状态。

使用 **Data -> Health** 确认覆盖率是否已刷新。如果覆盖率显示 **Not refreshed**，请在将覆盖率摘要视为当前值之前运行字段覆盖率或面板质量工作流。

## 6. Live 影子运行

Live 是影子/生产准备工作流。它准备目标持仓和手动审核订单；不连接券商 API 或提交订单。

```powershell
.\.venv\Scripts\python.exe scripts\check_live_readiness.py --config configs\live.local.yaml --universe cn_all
.\.venv\Scripts\python.exe scripts\run_live_superalpha.py --config configs\live.local.yaml --universe cn_all --superalpha-id superalpha_xxx --dry-run --skip-parity
```

使用 Dashboard **Live** 页面检查活跃的 Superalpha、数据就绪状态、最新持仓和订单审核状态。

## 状态处理手册

当 Dashboard 明确报告问题但你在决定下一步时使用此表。

| 状态 | 含义 | 下一步 |
| --- | --- | --- |
| preflight warning | 本地配置或环境检查发现风险设置，通常是 `configs\datasource.local.yaml` 中的密钥或缺少必需路径 | 将密钥移出本地 YAML，在 shell 中设置 `TUSHARE_TOKEN`，然后重新运行 `scripts\preflight_guard.py --strict` |
| coverage not refreshed | 目录覆盖率字段缺失或过期；Data Health 尚无法可靠判断字段覆盖率 | 刷新字段目录或运行现有面板质量工作流，再信任覆盖率摘要 |
| missing artifact | 运行、Live 输出或比较输入指向未生成或已被删除的产物 | 先重新打开生成该产物的运行；仅在产物状态为 complete 后才比较指标 |
| memory hard limit | 源分区内存保护在机器资源耗尽前停止了任务 | 降低 `max_eval` 或 `batch_size`，保持分区加载启用，或仅在机器有余量时提高硬限制 |
| stale lock | 闭环锁有旧的心跳或属于 Dashboard 外的进程 | 检查锁所有者和进程状态；让现有的闭环超时/清理路径处理锁 |

磁盘增长主要来自 `data/lake`、`data/duckdb` 和 `data/alpha_universe_store`。只读审计辅助工具是 `scripts/audit_analysis_artifacts.py`；Windows 上运行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_analysis_artifacts.py --store-root data\alpha_universe_store
```

审计报告分析产物大小和缺失输出；它不删除文件。使用现有的保留选项和运行健康摘要来决定哪些可以归档或清理。
