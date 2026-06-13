# Alpha Quarry

**闭环 Alpha 因子挖掘与研究平台 / Closed-loop Alpha Mining & Factor Research Platform for China A-Share Market**

[![CI](https://github.com/pengfeijiang320-eng/AlphaQuarry/actions/workflows/ci.yml/badge.svg)](https://github.com/pengfeijiang320-eng/AlphaQuarry/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性 (Features)

- **闭环挖掘** — 候选生成 → 评估 → 反馈 → 再挖掘的自动化闭环工作流
- **分层表达式** — 支持时序、截面、分组、逻辑、向量、回归等 84+ 运算符
- **因子研究** — IC 分析、分层收益、多空/纯多头组合、双重排序、样本内/外验证
- **实盘准备** — SuperAlpha 组合信号生成、目标持仓、订单审核（影子模式）
- **可视化面板** — React + FastAPI 本地 Dashboard，浏览因子库、PnL 曲线、Live 状态

## 闭环运行优化

闭环运行支持搜索空间扩展，避免多轮运行时候选重复:

- **Mutation**: 从高分因子片段生成新候选 (`enable_feedback_mutation: true`)
- **字段轮转**: 每轮聚焦不同字段子集做完整探索 (`field_rotation_focus_count: 3-5`)
- **Budget 轮转**: 不同轮次侧重不同层 (`budget_rotation_mode: round_robin`)

详见 [使用手册](alpha_mining/workflow/USAGE.md)。

## 架构概览 (Architecture)

```
Data Lake -> DuckDB -> Universe Store -> Closed Loop -> Analysis Run -> Factor Library -> Superalpha -> Live
```

详细数据流：

```
Tushare API → Parquet Lake → DuckDB Catalog
    → Closed-Loop Engine (candidate generation → evaluation → feedback)
    → Alpha Universe Store (per-universe results)
    → Dashboard (read-only browsing)
```

## 快速开始 (Quick Start)

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install -e ".[dev,viz]"
```

### 2. 配置数据源

```bash
# 复制配置模板
cp configs/datasource.example.yaml configs/datasource.local.yaml

# 设置 Tushare token（推荐环境变量方式）
export TUSHARE_TOKEN="your_token_here"
```

> **Tushare 积分要求**: 本项目使用 [Tushare](https://tushare.pro) 作为 A 股数据源。2000 积分即可运行核心流程（日线行情 + 涨跌停 + 停牌），5000 积分可解锁财务数据和资金流。详见 [Tushare 积分参考](docs/tushare_points_reference.md)。

### 3. 构建数据湖并运行闭环挖掘

```bash
# 预检 Tushare 权限（推荐，可确认当前积分可用的接口）
python scripts/preflight_tushare.py --config configs/datasource.local.yaml

# 构建数据湖（首次，--tier standard 适用于 2000 积分）
python scripts/bootstrap_tushare_lake.py --config configs/datasource.local.yaml --start-date 2016-01-01 --tier standard

# 运行闭环挖掘
python scripts/run_closed_loop.py --source-backend duckdb --duckdb-path data/duckdb/market.duckdb --source-view v_project_panel_cn_a --start-date 2022-01-01 --end-date 2026-04-21 --universe cn_all --request-new 10 --batch-size 5 --max-eval 120
```

### 4. 启动 Dashboard

```bash
# 安装前端
cd dashboard/frontend && npm install && npm run build && cd ../..

# 启动服务
python scripts/run_factor_dashboard.py --store-root data/alpha_universe_store --host 127.0.0.1 --port 8008
```

打开 `http://127.0.0.1:8008` 查看。请保持 Dashboard 绑定在 `127.0.0.1`；不要暴露到局域网、VPN 或公网。

## 文档 (Documentation)

- [快速上手](docs/quickstart.md)
- [详细使用手册](PROJECT_USER_MANUAL.md)
- [运算符参考](docs/operator_registry_audit.md) — 84/84 运算符覆盖
- [字段目录](docs/field_catalog.md) — 数据字段与因子族
- [实验产物规范](docs/experiment_artifacts_schema.md)
- [Live 实盘手册](docs/live_superalpha_runbook.md)
- [维护检查清单](docs/maintenance.md)

## 技术栈 (Tech Stack)

| 层 | 技术 |
|----|------|
| 数据采集 | Tushare |
| 数据存储 | Parquet Lake + DuckDB |
| 后端引擎 | Python 3.10+, pandas, numpy, scipy |
| Dashboard API | FastAPI + Uvicorn |
| Dashboard 前端 | React 19 + TypeScript + Vite + ECharts |
| 测试 | pytest |

## 测试 (Testing)

```bash
python -m pytest alpha_mining/tests tests -q
```

## 许可证 (License)

[MIT](LICENSE)

## 免责声明 (Disclaimer)

**本软件仅供研究和教育用途，不适用于生产环境交易或投资决策。**

- Alpha Quarry 是一个用于探索和测试 Alpha 因子想法的量化研究平台
- 本平台不提供投资建议，也不保证任何投资回报
- 任何策略或因子的过往表现不保证未来结果
- 用户在做出投资决策前应自行进行尽职调查
- 作者和贡献者不对因使用本软件而产生的任何财务损失负责
- 将本软件用于实盘交易需要适当的风险管理和合规性

**数据来源声明**: 本项目使用 [Tushare](https://tushare.pro) 作为数据源。使用本项目即表示您同意遵守 Tushare 的使用条款。本项目不存储或分发任何市场数据，所有数据通过 Tushare API 实时获取。

---

**贡献指南**: 欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献规范。
