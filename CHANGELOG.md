# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CI/CD pipeline with GitHub Actions (lint + test + frontend build)
- SECURITY.md with vulnerability reporting guidelines
- CONTRIBUTING.md with development setup and contribution guidelines
- CHANGELOG.md for tracking version changes

## [0.1.0-alpha] - 2026-06-12

### Added

#### Core Mining Engine
- 84+ expression operators organized by category:
  - Time series operators (ts_mean, ts_std, ts_rank, ts_delta, etc.)
  - Cross-sectional operators (cs_rank, cs_zscore, etc.)
  - Group operators (group_mean, group_std, etc.)
  - Logical operators (where, and, or, not, etc.)
  - Vector operators (vector_add, vector_multiply, etc.)
  - Regression operators (reg_beta, reg_resid, etc.)
- Expression parser with function-style syntax
- Operator registry with decorator-based registration
- PanelStore for efficient date x code panel management

#### Closed-Loop Workflow
- Automated candidate generation → evaluation → feedback → mutation loop
- Layered search space (L0-L4) with configurable budgets
- Fragment mutation (operator swap, window change, crossover)
- Frequency ratio feedback mechanism
- Adaptive exploration ratio

#### Factor Research Library
- IC (Information Coefficient) analysis
- Layer analysis with configurable quantiles
- Long-short and long-only portfolio construction
- Double sort with Newey-West statistics
- Train/validation/OOS sample split analysis
- Factor effectiveness scoring and correlation filtering
- Turnover analysis

#### Data Architecture
- Tushare data integration with tiered preflight checks
- Parquet Lake for raw and curated data storage
- DuckDB catalog for efficient querying
- Incremental data update support

#### Dashboard
- FastAPI backend with RESTful API
- React 19 + TypeScript frontend
- ECharts for interactive visualizations
- TanStack Table for data browsing
- Factor library browser
- PnL curve visualization
- Live mode status display

#### SuperAlpha
- Shadow mode for live signal generation
- Target portfolio construction
- Order review workflow

#### Reproducibility
- Snapshot mechanism for input data
- Expression hash-based deduplication
- Structural hash for AST-level comparison
- Field catalog documentation

#### CLI Tools
- `run_closed_loop.py` - Main closed-loop mining script
- `bootstrap_tushare_lake.py` - Initial data lake construction
- `update_tushare_lake.py` - Incremental data updates
- `build_duckdb_catalog.py` - DuckDB catalog rebuilding
- `run_factor_dashboard.py` - Dashboard server
- `preflight_tushare.py` - Tushare permission checker
- `preflight_guard.py` - Pre-commit security guard

### Documentation
- Quick start guide
- Operator registry audit (84/84 coverage)
- Field catalog
- Experiment artifacts schema
- Closed-loop optimization report
- Live trading runbook
- Maintenance checklist
- Tushare points reference

[Unreleased]: https://github.com/pengfeijiang320-eng/AlphaQuarry/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/pengfeijiang320-eng/AlphaQuarry/releases/tag/v0.1.0-alpha
