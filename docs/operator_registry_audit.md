# 运算符注册表参考

最后核对本地代码注册表：2026-05-09。

## 摘要

- 本地注册运算符：**84**
- 签名覆盖运算符：**84**
- 缺失签名：**无**
- 已签名运算符缺失实现：**无**
- 运算符真相来源：
  - 实现：`alpha_mining/operators/`
  - 注册表连接：`alpha_mining/registry.py`
  - 类型签名：`alpha_mining/mining/operator_signatures.py`

## 搜索与安全说明

- 候选表达式经过 parser、运算符注册表、签名注册表、泄露字段过滤器、规范化器和预筛选器检查。
- `hump` 和 `trade_when_hold` 是有状态/高风险包装器，除非显式启用，否则由 `enable_stateful_phase2_ops=false` 控制。
- `zero_like(x)` 是项目扩展，用于生成索引对齐的零序列而不创建自减表达式。
- `trade_when` 是无状态的 `if_else` 风格。有状态的持有直到退出行为隔离在 `trade_when_hold` 中。
- `ts_regression` 是简化的滚动回归变体，不是完整的 WQ lag/rettype 表面。

## 已注册运算符

| 运算符 | 签名 |
|---|---|
| `abs` | `(scalar)->scalar` |
| `add` | `(scalar,scalar)->scalar` |
| `bucket` | `(scalar,literal)->group` |
| `cs_quantile` | `(scalar)->scalar`; `(scalar,literal)->scalar`; `(scalar,literal,literal)->scalar` |
| `days_from_last_change` | `(scalar)->scalar` |
| `densify` | `(group)->group` |
| `div` | `(scalar,scalar)->scalar` |
| `divide` | `(scalar,scalar)->scalar` |
| `equal` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `event_active` | `(bool,scalar)->scalar` |
| `event_decay` | `(bool,scalar)->scalar`; `(bool,scalar,literal)->scalar` |
| `greater` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `greater_equal` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `group_cartesian_product` | `(group,group)->group` |
| `group_mean` | `(scalar,group)->scalar` |
| `group_median` | `(scalar,group)->scalar` |
| `group_neutralize` | `(scalar,group)->scalar` |
| `group_normalize` | `(scalar,group)->scalar` |
| `group_rank` | `(scalar,group)->scalar` |
| `group_scale` | `(scalar,group)->scalar` |
| `group_sum` | `(scalar,group)->scalar` |
| `group_zscore` | `(scalar,group)->scalar` |
| `hump` | `(scalar)->scalar`; `(scalar,literal)->scalar` |
| `if_else` | `(bool,scalar,scalar)->scalar` |
| `inverse` | `(scalar)->scalar` |
| `is_nan` | `(scalar)->bool` |
| `is_not_nan` | `(scalar)->bool` |
| `left_tail` | `(scalar)->scalar`; `(scalar,literal)->scalar` |
| `less` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `less_equal` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `log` | `(scalar)->scalar` |
| `max` | `(scalar,scalar)->scalar` |
| `min` | `(scalar,scalar)->scalar` |
| `mul` | `(scalar,scalar)->scalar` |
| `multiply` | `(scalar,scalar)->scalar` |
| `normalize` | `(scalar)->scalar` |
| `not_equal` | `(scalar,scalar)->bool`; `(scalar,literal)->bool`; `(literal,scalar)->bool` |
| `power` | `(scalar,literal)->scalar`; `(scalar,scalar)->scalar` |
| `quantile` | `(scalar)->scalar`; `(scalar,literal)->scalar`; `(scalar,literal,literal)->scalar` |
| `rank` | `(scalar)->scalar` |
| `regression_neut` | `(scalar,scalar)->scalar` |
| `reverse` | `(scalar)->scalar` |
| `right_tail` | `(scalar)->scalar`; `(scalar,literal)->scalar` |
| `s_log_1p` | `(scalar)->scalar` |
| `scale` | `(scalar)->scalar`; `(scalar,literal)->scalar`; `(scalar,literal,literal,literal)->scalar` |
| `sign` | `(scalar)->scalar` |
| `signed_power` | `(scalar,literal)->scalar`; `(scalar,scalar)->scalar` |
| `sqrt` | `(scalar)->scalar` |
| `sub` | `(scalar,scalar)->scalar` |
| `subtract` | `(scalar,scalar)->scalar` |
| `trade_when` | `(bool,scalar,scalar)->scalar` |
| `trade_when_hold` | `(bool,scalar,bool)->scalar` |
| `truncate` | `(scalar)->scalar`; `(scalar,literal)->scalar` |
| `ts_arg_max` | `(scalar,window)->scalar` |
| `ts_arg_min` | `(scalar,window)->scalar` |
| `ts_av_diff` | `(scalar,window)->scalar` |
| `ts_backfill` | `(scalar,window)->scalar` |
| `ts_corr` | `(scalar,scalar,window)->scalar` |
| `ts_count_nans` | `(scalar,window)->scalar` |
| `ts_covariance` | `(scalar,scalar,window)->scalar` |
| `ts_decay_exp_window` | `(scalar,window)->scalar`; `(scalar,window,literal)->scalar` |
| `ts_decay_linear` | `(scalar,window)->scalar` |
| `ts_delay` | `(scalar,window)->scalar` |
| `ts_delta` | `(scalar,window)->scalar` |
| `ts_ir` | `(scalar,window)->scalar` |
| `ts_max` | `(scalar,window)->scalar` |
| `ts_mean` | `(scalar,window)->scalar` |
| `ts_median` | `(scalar,window)->scalar` |
| `ts_min` | `(scalar,window)->scalar` |
| `ts_product` | `(scalar,window)->scalar` |
| `ts_rank` | `(scalar,window)->scalar` |
| `ts_regression` | `(scalar,scalar,window)->scalar`; `(scalar,scalar,window,literal)->scalar` |
| `ts_std_dev` | `(scalar,window)->scalar` |
| `ts_sum` | `(scalar,window)->scalar` |
| `ts_zscore` | `(scalar,window)->scalar` |
| `vec_avg` | `(vector)->scalar` |
| `vec_count` | `(vector)->scalar` |
| `vec_max` | `(vector)->scalar` |
| `vec_min` | `(vector)->scalar` |
| `vec_stddev` | `(vector)->scalar` |
| `vec_sum` | `(vector)->scalar` |
| `winsorize` | `(scalar,literal)->scalar` |
| `zero_like` | `(scalar)->scalar` |
| `zscore` | `(scalar)->scalar` |

## 审计命令

```powershell
.\.venv\Scripts\python.exe scripts\check_operator_registry.py
```

此命令写入 `artifacts/dev/operator_signature_audit.csv`，如果实现/签名覆盖不一致则失败。
