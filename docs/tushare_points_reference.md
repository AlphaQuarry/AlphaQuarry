# Tushare 积分参考

本项目使用 [Tushare](https://tushare.pro) 作为 A 股数据源。积分门槛基于 [Tushare 官方文档](https://tushare.pro/document/2?doc_id=108)。

## 积分三档

项目使用的全部接口分为三个积分档：

### 2000 积分 — 核心数据

| 接口 | 用途 | 数据组 |
|------|------|--------|
| `stock_basic` | 股票列表 | P0 dim |
| `trade_cal` | 交易日历 | P0 dim |
| `index_classify` | 申万行业分类 | P0 dim |
| `index_member_all` | 行业成分 | P0 dim |
| `namechange` | 股票曾用名 | P1 dim |
| `daily` | 日线行情 (OHLCV) | P0 |
| `daily_basic` | 每日指标 (PE/PB/换手率/市值) | P0 |
| `adj_factor` | 复权因子 | P0 |
| `stk_limit` | 涨跌停价格 | P1 |
| `suspend_d` | 停复牌信息 | P1 |
| `moneyflow` | 个股资金流向 | P3 |
| `moneyflow_ths` | 同花顺资金流 | P3_legacy |
| `index_daily` | 指数日线行情 | index |
| `index_weight` | 指数成分权重 | index |

> **运行闭环挖掘的最低要求**: 2000 积分即可使用 `--tier basic` 或 `--tier standard`。

### 5000 积分 — 财务数据 + 集合竞价

| 接口 | 用途 | 数据组 | 说明 |
|------|------|--------|------|
| `income_vip` | 利润表 (VIP全市场) | P2 | 非VIP版2000积分，VIP全市场拉取需5000 |
| `balancesheet_vip` | 资产负债表 (VIP) | P2 | 同上 |
| `cashflow_vip` | 现金流量表 (VIP) | P2 | 同上 |
| `fina_indicator_vip` | 财务指标 (VIP) | P2 | 同上 |
| `stk_auction_o` | 开盘集合竞价 | P4_auction | |
| `stk_auction_c` | 收盘集合竞价 | P4_auction | |

> **说明**: 项目中财务数据使用 `_vip` 接口按时间范围拉取全市场数据，需要 5000 积分。非 VIP 版（按单只股票拉取）2000 积分即可，但不适用于本项目的批量拉取模式。

### 10000 积分 — 特色数据

| 接口 | 用途 | 数据组 |
|------|------|--------|
| `cyq_perf` | 每日筹码及胜率 | P4 |
| `cyq_chips` | 每日筹码分布 | P4 |
| `stk_factor_pro` | 技术面因子 | P4 |
| `report_rc` | 券商盈利预测 | P5 |

## 数据层级 (--tier)

| Tier | 包含数据组 | 推荐积分 | 说明 |
|------|-----------|----------|------|
| basic | P0 | 2000 | 日线+复权+每日指标 |
| standard | P0+P1 | 2000 | +涨跌停+停牌 |
| extended | P0+P1+P2 | 5000 | +VIP财务数据 |
| full | 全部 | 10000 | +筹码/技术因子/券商预测 |

## 预检脚本

运行以下命令可快速检测当前积分档（仅需 3 次 API 调用）：

```bash
python scripts/preflight_tushare.py --config configs/datasource.local.yaml
```

输出示例：
```
[preflight] Tushare 权限预检
============================================================
📊 探测结果 (3 次 API 调用):
  ✅ daily                     2000 积分  可用 (5000 行)
  ❌ stk_auction_o             5000 积分  权限不足
  ❌ cyq_perf                 10000 积分  权限不足

============================================================
判定积分档: ≥2000
建议: --tier standard
```

## 如何查看/升级积分

- 登录 [Tushare Pro](https://tushare.pro) → "个人中心" → "积分"
- 完善信息、邀请好友、或购买积分包
