# 维护检查清单

本检查清单面向本地单机研究工作站。仅使用现有 Dashboard 页面、脚本和产物，不引入清理自动化或新任务。

## 每日检查

- 检查 **Overview freshness**：字段目录新鲜度、最新分析运行时间和 Live 最新状态
- 检查 **preflight**：从 Overview 或运行 `scripts/preflight_guard.py --strict`（在数据刷新或 Live 准备之前）
- 检查 **Closed Loop**：运行中/失败的任务。失败任务请检查任务状态、失败提示、stderr 尾部、锁所有者和最近的 `run_health`
- 检查 **Live readiness**：在使用影子持仓或手动审核订单之前

## 每周检查

- 检查 **Data Health coverage**。如果覆盖率显示 **Not refreshed**，在将覆盖率摘要视为当前值之前运行现有刷新或质量工作流
- 检查 `run_health`：硬限制事件、内存警告和重复失败的闭环运行
- 运行只读产物审计 `scripts/audit_analysis_artifacts.py`：检查分析产物增长和缺失输出
- 检查 `data/lake`、`data/duckdb`、`data/alpha_universe_store` 的磁盘增长

## 预检失败

- 将 Tushare 密钥保存在 shell 环境中，不要放在 `configs/datasource.local.yaml`
- 如果预检报告本地 token 非空，手动清除 `tushare.token` 并在 shell 中设置 `TUSHARE_TOKEN`
- 不要在 bug 报告、日志、截图或提交的配置文件中暴露 token 值
- 自定义 `tushare.http_url` 是信息性的；确认它是私有且预期的

## 磁盘增长

- `data/lake`：随原始表、curated 表、快照和字段目录数据增长
- `data/duckdb`：随本地 DuckDB 目录/查询层增长
- `data/alpha_universe_store`：随 universe 工作区、闭环候选、分析运行、因子库记录、SuperAlpha 产物、Live 影子产物和 Dashboard 任务日志增长
- 优先只读检查。使用保留设置和现有审计输出来决定哪些可以归档或删除

## 闭环故障排查

- 从 Dashboard 中的任务状态和失败提示开始
- 修改参数前先读取 stderr 尾部
- 当任务报告锁冲突或过期锁风险时检查锁所有者
- 检查 `run_health`：内存硬限制事件、DuckDB 临时存储压力、数据为空或重复分析失败
- **Running outside dashboard**：在手动操作前确认进程是否仍在运行
- **Interrupted**：在重启同一 universe 前检查最后的 stderr 尾部和运行产物

## 测试选择

- Dashboard 后端变更后运行 Dashboard API 测试：`.\.venv\Scripts\python.exe -m pytest tests\test_dashboard_next_workbench_api.py tests\test_dashboard_workbench_api.py tests\test_factor_dashboard_api.py tests\test_dashboard_live_api.py tests\test_superalpha_dashboard_api.py -q`
- 闭环、SuperAlpha、因子库、挖掘或工作流变更后运行 `alpha_mining/tests`
- 维护批次完成前运行全量测试：`.\.venv\Scripts\python.exe -m pytest -q`
- 前端或共享 Dashboard 类型/复制变更后在 `dashboard/frontend` 运行 `npm run build`

## 状态术语

- **Missing artifact**：预期文件不存在或不可读
- **Partial metrics**：运行有部分比较指标但不是完整指标集
- **Not refreshed**：覆盖率或新鲜度元数据不存在/过期，不一定是源数据有问题
- **Interrupted**：Dashboard 恢复了运行中的任务记录但进程已不存在
- **Running outside dashboard**：Dashboard 重启后进程仍在运行，但不在当前 FastAPI 进程内存中跟踪
