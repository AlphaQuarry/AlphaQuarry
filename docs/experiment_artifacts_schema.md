# 实验产物规范

本规范记录闭环实验产物写入的最小稳定元数据。它有意保持精简：记录可复现性输入，不引入新的缓存或记录服务。

## 输入清单

输入清单写入 universe 工作区，按 `manifest_id` 键控。

必需的高层字段：

- `manifest_schema_version`：当前 schema 版本，默认 `v2`
- `source_backend`、`duckdb_path`、`source_view`、`source_path`、`snapshot_path`：数据来源
- `date_col`、`code_col`、`group_fields`、`vector_fields`、`base_frame_cols`：面板布局
- `date_range`、`field_catalog_version`、`run_filters`、`search_mode`：数据集和候选规划上下文
- `field_preprocessing_config`：原始字段预处理策略。默认标量包装器为 `winsorize(ts_backfill(field, 120), 4.0)`
- `simulation_config`：完整的 `AlphaSimulationConfig`，包括组合构建模式
- `moneyflow_source`：默认 `moneyflow`；`moneyflow_ths` 作为 legacy 源仍可用
- `operator_registry`：已实现运算符名称和数量
- `signature_registry`：已签名运算符名称和数量
- `columns`、`dtypes`、`row_count`：实际加载的面板形状

## 表达式注册表

每个选中的表达式行记录：

- `simulation_config_json`
- `input_manifest_id`
- `input_source_path`
- `panel_signature_hash`
- `search_mode`

候选生成元数据与旧运行兼容，可用时可能包含 source、family、layer、mutation、fragment、pair、structural hash 和 template 字段。

## 兼容性

缺少这些字段的旧清单仍可读取。新字段是增量添加的，下游读取器应将其视为可选。
