# 输入数据格式说明

## 必备字段

- `trade_date`：交易日期，建议 `YYYY-MM-DD`。
- `znz_code`：股票代码（如 `000001.SZ`）。
- `pct_chg`：当期收益率（小数形式，`0.01` 表示 1%）。
- `circ_mv`：流通市值（正数，用于对数中性化）。
- 因子列：任意数量的数值列（如 `factor_value`、`factor_quality`）。

## 字段约束

- 每个 `(trade_date, znz_code)` 最好唯一。
- `circ_mv` 必须大于 0（否则中性化会被过滤）。
- 因子列和收益列建议为 `float`。
- 同一交易日应有足够股票数量（至少 2 条，分层建议大于层数）。

## 最小示例数据

文件：`examples/sample_factor_data.csv`

可直接用于 notebook：

```python
import pandas as pd
from factor_alalyze_lib import *

df = pd.read_csv("examples/sample_factor_data.csv")
df["trade_date"] = pd.to_datetime(df["trade_date"])

factor_cols = ["factor_value", "factor_quality", "factor_momentum"]

df = process_future_return(df, return_col="pct_chg", period=5)
df_processed = process_factor_data(df, factor_cols, market_value_column="circ_mv", is_timeseries=True)
ic_df, summary_df = calculate_icir(df_processed, factor_cols, return_col="pct_chg", period=5)
layer_results = factor_layer_analysis(df_processed, factor_cols, return_col="pct_chg", period=5, layers=5)
```
