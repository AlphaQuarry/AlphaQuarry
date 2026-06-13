# 项目字段目录参考

最后对照 `data/lake/meta/field_catalog.parquet` 和当前目录构建代码检查于 2026-05-09。

如需查看 Tushare 接口和本地字段的电子表格视图，请参阅 `docs/tushare_interface_fields.csv`。

## 概述

本文档引用的旧版本本地字段目录产物在较新的可选表重建之前包含 **86 个字段**：

- `SCALAR`：67 个
- `EVENT`：14 个
- `GROUP`：3 个
- `ID`：2 个

当前代码在重建 DuckDB 目录时还可以添加最新的闭环分析列：

- `can_trade`
- `can_buy`
- `can_sell`
- `is_one_price_up_limit`
- `is_one_price_down_limit`
- `is_limit_up_close`
- `is_limit_down_close`
- 字段目录中的 `factor_family`
- 来自 Tushare `cyq_chips` 的 `cyq_chip_*` 每日衍生特征

如果现有本地 `field_catalog.parquet` 中缺少这些列，请重建或刷新 DuckDB 目录。旧版目录文件仍然可读；运行时字段族推断会回退到字段名称和类别。

## 搜索规则

- `SCALAR` 字段如果可搜索且不是泄露/审计字段，则可以进入因子表达式。
- `GROUP` 字段用作分组参数，例如 `group_rank(close, industry)`。
- `EVENT`、`ID`、收益率、审计、过滤和元数据字段不应添加到 `--include-fields`。
- `--include-fields` 控制标量/向量搜索字段。
- `--group-fields` 控制分组字段。
- 方向性可交易字段仅用于过滤/分析，不用于表达式搜索。

推荐的安全基础字段池：

```powershell
--group-fields industry,sector `
--include-fields close,open,high,low,volume,amount,circ_mv,total_mv
```

## 因子族

候选生成和排名使用四个机构因子族：

- `price_volume`
- `fundamental`
- `moneyflow`
- `analyst`

目录构建器会为刷新的目录写入 `factor_family`。候选产物还包括：

- `factor_family`
- `factor_family_mix_json`
- `primary_factor_family`

## 字段覆盖率

字段覆盖率会显式刷新并存储在 `data/lake/meta/field_catalog.parquet` 中。
默认范围是闭环数据源范围：配置的 `source_view`、配置的 `run_filters`，以及传递给刷新脚本的日期范围。

覆盖率列：

- `coverage_scope`：当前为 `closed_loop`
- `coverage_start_date` / `coverage_end_date`：请求的统计窗口
- `coverage_row_count`：日期和运行过滤后的分母
- `non_null_count`：该字段的非空观测值
- `coverage_rate` / `missing_rate`：非空比率和缺失比率
- `finite_count` / `finite_rate`：有限数值观测值；非数值字段为空
- `coverage_status`：`ok`、`missing_field` 或 `skipped_heavy_source`
- `coverage_updated_at_utc`：刷新时间戳

默认情况下，刷新脚本使用快速安全范围：市场热门表字段精确计算，而来自 `v_project_financial_asof_daily` 的财务/as-of 字段标记为 `skipped_heavy_source`。这避免了在常规字段筛选期间强制执行昂贵的每日 as-of 连接。仅在明确想要运行较重的财务/as-of 覆盖率任务时才传递 `--include-heavy-asof`。

刷新 2016-2024 年闭环字段覆盖率并写回目录：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_coverage.py `
  --config configs\datasource.local.yaml `
  --start-date 2016-01-01 `
  --end-date 2024-12-31 `
  --duckdb-memory-limit 4GB `
  --duckdb-threads 2 `
  --duckdb-max-temp-directory-size 12GB `
  --update-field-catalog `
  --output-csv artifacts\data_quality\field_coverage_2016_2024.csv
```

刷新脚本默认将 DuckDB 临时文件放在 `<duckdb-path>.tmp` 下的隔离运行目录中，例如 `data\duckdb\market.duckdb.tmp\run_field_coverage_YYYYMMDD_HHMMSS`，并在运行前后清理该隔离目录。这使溢出文件不会占用 Windows 用户临时目录（`C:`），并使中断的运行更易于清理。仅在需要检查 DuckDB 溢出文件时才传递 `--no-cleanup-duckdb-temp`。

需要时单独运行财务/as-of 覆盖率：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_coverage.py `
  --config configs\datasource.local.yaml `
  --start-date 2016-01-01 `
  --end-date 2024-12-31 `
  --include-heavy-asof `
  --include-source-tables v_project_financial_asof_daily `
  --duckdb-memory-limit 4GB `
  --duckdb-threads 1 `
  --duckdb-max-temp-directory-size 20GB `
  --output-csv artifacts\data_quality\field_coverage_finance_asof_2016_2024.csv
```

快速检查稀疏字段：

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_parquet('data/lake/meta/field_catalog.parquet'); print(df[['field_name','category','factor_family','coverage_rate','missing_rate','non_null_count']].sort_values('coverage_rate').to_string(index=False))"
```

## 当前本地目录中的字段分组

### ID

`code`, `date`

### GROUP

`industry`, `sector`, `subindustry`

### 价格

`close`, `high`, `low`, `open`

### 流动性

`amount`, `volume`

### 估值 / 市值

`circ_mv`, `total_mv`

### 收益率 / 仅用于分析

`pct_chg`, `ret_1d`

不要将这些放入表达式搜索。它们用于分析和诊断。

### 涨跌停 / 上市 / 交易状态

当前本地目录：

`days_since_listed`, `down_limit`, `is_st`, `is_suspended`, `up_limit`, `tradable`, `universe`

目录重建后当前代码支持：

`can_trade`, `can_buy`, `can_sell`, `is_one_price_up_limit`, `is_one_price_down_limit`, `is_limit_up_close`, `is_limit_down_close`

这些是过滤/分析字段。`can_buy` 和 `can_sell` 在启用时驱动方向性多头可交易约束。

### 资金流

默认资金流来源是 Tushare `moneyflow`（`doc_id=170`）。项目仅保留官方金额字段：

`moneyflow_buy_sm_amount`, `moneyflow_sell_sm_amount`, `moneyflow_buy_md_amount`, `moneyflow_sell_md_amount`, `moneyflow_buy_lg_amount`, `moneyflow_sell_lg_amount`, `moneyflow_buy_elg_amount`, `moneyflow_sell_elg_amount`, `moneyflow_net_mf_amount`

旧版 THS 字段如 `moneyflow_net_amount`、`moneyflow_net_d5_amount` 和 `moneyflow_*_amount_rate` 在 DuckDB/目录重建后不属于默认项目面板。

示例：

```powershell
--include-fields close,volume,amount,moneyflow_net_mf_amount,moneyflow_buy_sm_amount,moneyflow_sell_sm_amount,moneyflow_buy_lg_amount,moneyflow_sell_lg_amount,moneyflow_buy_elg_amount,moneyflow_sell_elg_amount
```

### 基本面 / 财务 As-Of

数值财务字段：

财务数据来自低频季度/报告事实，而非每日观测。数据湖按 `code` + `ann_date` + `end_date` 存储一行；DuckDB 通过公告日期 as-of 字段将其暴露给每日研究面板。当前项目获取 `income_vip`、`balancesheet_vip`、`cashflow_vip` 和 `fina_indicator_vip`；完整的接口到字段映射请参阅 `docs/tushare_interface_fields.csv`。

常用数值财务字段：

`fin_total_revenue`, `fin_revenue`, `fin_n_income_attr_p`, `fin_total_assets`, `fin_total_liab`, `fin_total_hldr_eqy_exc_min_int`, `fin_n_cashflow_act`, `fin_current_ratio`, `fin_quick_ratio`, `fin_cash_ratio`, `fin_assets_turn`, `fin_debt_to_assets`, `fin_roe`, `fin_roe_waa`, `fin_roe_dt`, `fin_roa`, `fin_roic`, `fin_q_roe`, `fin_cfps_yoy`, `fin_op_yoy`, `fin_ebt_yoy`, `fin_ocf_yoy`

财务公告/报告日期审计字段：

`fin_balance_ann_date`, `fin_balance_end_date`, `fin_cashflow_ann_date`, `fin_cashflow_end_date`, `fin_income_ann_date`, `fin_income_end_date`, `fin_indicator_ann_date`, `fin_indicator_end_date`

日期字段是审计/as-of 字段，不应进入表达式搜索。闭环 DuckDB 加载仅为请求的 `fin_*` 字段计算财务 as-of 连接；常规非财务运行继续使用精简热门表。

### 分析师

`report_rc_eps_mean`

### 筹码 / 技术

`cyq_perf` 字段：

`cyq_his_low`, `cyq_his_high`, `cyq_cost_5pct`, `cyq_cost_15pct`, `cyq_cost_50pct`, `cyq_cost_85pct`, `cyq_cost_95pct`, `cyq_weight_avg`, `cyq_winner_rate`

`cyq_chips` 字段：

完整的 Tushare `cyq_chips` 价格分布作为长事实表存储在 `facts/cyq_chips` 下，包含 `chip_price`、`chip_percent_raw_pct` 和标准化的 `chip_percent`。由于分布每个 `code,date` 有多行，项目面板仅暴露来自 `facts/cyq_chips_daily` 的每日衍生字段：

`cyq_chip_price_count`, `cyq_chip_percent_sum`, `cyq_chip_price_min`, `cyq_chip_price_max`, `cyq_chip_mode_price`, `cyq_chip_mode_percent`, `cyq_chip_weight_avg_price`, `cyq_chip_price_std`, `cyq_chip_cost_10pct`, `cyq_chip_cost_25pct`, `cyq_chip_cost_50pct`, `cyq_chip_cost_75pct`, `cyq_chip_cost_90pct`

这些衍生字段归类为 `chip`，当 chip 类别启用时可以进入表达式搜索。原始长表保留用于审计和自定义研究，但不会直接展开到每日面板中。

当前保留在项目面板中的 `stk_factor_pro` 字段：

项目现在从 Tushare `stk_factor_pro` 请求仅复权价格技术指标，并将其暴露为 `tech_*` 字段，例如 `tech_asi_qfq`、`tech_atr_qfq`、`tech_boll_mid_qfq`、`tech_macd_qfq`、`tech_rsi_qfq_6`、`tech_rsi_qfq_24` 和 `tech_xsii_td4_qfq`，以及非价格调整的 `tech_updays`、`tech_downdays`、`tech_topdays`、`tech_lowdays`。完整列表请参阅 `docs/tushare_interface_fields.csv`。

原始 Tushare `stk_factor_pro` 提供 `_bfq`、`_qfq` 和 `_hfq` 变体。本项目故意仅为技术因子保留 qfq 变体，这样面板不会为同一指标暴露多个调整变体。

`auction_o_open` 和 `auction_c_close` 是旧版可选字段，不再包含在默认 `p4` 更新组或项目面板中。

### 其他可选市场字段

`adj_factor`, `bfq_close`, `bfq_high`, `bfq_low`, `bfq_open`, `circ_mv_raw_wan`, `delist_date`, `dv_ratio`, `dv_ttm`, `hfq_close`, `hfq_high`, `hfq_low`, `hfq_open`, `limit_pre_close`, `list_date`, `list_status`, `market`, `pb`, `pe`, `pe_ttm`, `ps`, `ps_ttm`, `qfq_close`, `qfq_high`, `qfq_low`, `qfq_open`, `security_name`, `total_mv_raw_wan`, `turnover_rate`, `turnover_rate_f`, `volume_ratio`

谨慎使用这些字段。除非实验需要特定的原始或调整价格路径，否则优先使用规范调整的 `open/high/low/close`、标准化的 `amount/volume` 和标准化的 `circ_mv/total_mv`。

## 常用字段池

价量：

```powershell
--include-fields close,open,high,low,volume,amount,circ_mv,total_mv
```

资金流：

```powershell
--include-fields close,volume,amount,moneyflow_net_mf_amount,moneyflow_buy_sm_amount,moneyflow_sell_sm_amount,moneyflow_buy_md_amount,moneyflow_sell_md_amount,moneyflow_buy_lg_amount,moneyflow_sell_lg_amount,moneyflow_buy_elg_amount,moneyflow_sell_elg_amount
```

基本面：

```powershell
--include-fields close,circ_mv,total_mv,fin_roe,fin_roa,fin_grossprofit_margin,fin_total_assets,fin_total_liab,fin_n_income_attr_p
```

分析师：

```powershell
--include-fields close,circ_mv,total_mv,report_rc_eps_mean
```

## 刷新命令

重建 DuckDB 目录和字段目录：

```powershell
.\.venv\Scripts\python.exe scripts\build_duckdb_catalog.py --config configs\datasource.local.yaml
```

仅从现有 DuckDB 目录刷新字段目录：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_catalog.py --config configs\datasource.local.yaml
```

刷新字段覆盖率并合并到现有字段目录：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_field_coverage.py `
  --config configs\datasource.local.yaml `
  --start-date 2016-01-01 `
  --end-date 2024-12-31 `
  --duckdb-max-temp-directory-size 12GB `
  --update-field-catalog
```

检查当前本地目录：

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_parquet('data/lake/meta/field_catalog.parquet'); print(df[['field_name','field_type','category','is_searchable']].to_string())"
```
