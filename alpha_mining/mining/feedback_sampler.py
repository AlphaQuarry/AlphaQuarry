from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..ast_nodes import ExpressionNode, FieldNode, FunctionCallNode, LiteralNode
from ..parser import ExpressionParseError, parse_expression


@dataclass(frozen=True)
class FeedbackSamplerConfig:
    enabled: bool = True
    exploit_ratio: float = 0.55
    min_explore_ratio: float = 0.30
    lookback_batches: int = 50
    cooldown_lookback: int = 10
    min_all_count: int = 5  # 出现率比值的最小样本量


class FeedbackSampler:
    """
    First-pass feedback sampler facade.

    The interface is intentionally stable while the initial behavior is
    conservative: it summarizes historical winners and exposes weight hints
    without forcing candidate generation to depend on them.
    """

    def __init__(self, config: FeedbackSamplerConfig | None = None) -> None:
        self.config = config or FeedbackSamplerConfig()

    def build_weight_hints(
        self,
        scoreboard_df: pd.DataFrame | None = None,
        expression_registry_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return _empty_hints(enabled=False)
        if scoreboard_df is None or scoreboard_df.empty:
            hints = _empty_hints(enabled=True)
            hints["reason"] = "no_scoreboard"
            return hints
        work = scoreboard_df.copy()

        # 滑动窗口：按 submitted_at_utc 过滤最近 lookback_batches 个批次
        if self.config.lookback_batches > 0 and "submitted_at_utc" in work.columns:
            work["submitted_at_utc"] = pd.to_datetime(work["submitted_at_utc"], errors="coerce")
            work = work.sort_values("submitted_at_utc", ascending=False)
            if "analysis_run_id" in work.columns:
                unique_runs = work["analysis_run_id"].dropna().unique()
                recent_runs = unique_runs[: self.config.lookback_batches]
                work = work[work["analysis_run_id"].isin(recent_runs)]
            else:
                work = work.head(self.config.lookback_batches * 5)

        score_col = _select_feedback_score_column(work)
        if score_col not in work.columns:
            hints = _empty_hints(enabled=True)
            hints["reason"] = "missing_score"
            return hints
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        ordered = work.sort_values(score_col, ascending=False)
        top = ordered.head(max(1, int(len(work) * 0.2)))
        negative = ordered[ordered[score_col].fillna(0.0) <= 0.0].tail(max(1, int(len(work) * 0.2)))
        if "turnover_long_only_mean" in ordered.columns:
            turnover = pd.to_numeric(ordered["turnover_long_only_mean"], errors="coerce")
            negative = pd.concat([negative, ordered[turnover >= 1.0]], axis=0).drop_duplicates()

        min_count = int(self.config.min_all_count)

        return {
            "enabled": True,
            "score_col": score_col,
            "score_basis": _score_basis(score_col),
            "top_count": int(len(top)),
            "negative_count": int(len(negative)),
            "field_weights": _compute_frequency_ratio(work.get("fields"), top.get("fields"), min_all_count=min_count),
            "operator_weights": _compute_frequency_ratio(
                work.get("operators"), top.get("operators"), min_all_count=min_count
            ),
            "substructure_weights": _compute_substructure_frequency_ratio(
                work.get("expression"),
                top.get("expression"),
                min_all_count=min_count,
            ),
            "family_weights": _count_csv_values(top.get("family")),
            "factor_family_weights": _compute_frequency_ratio(
                work.get("factor_family"),
                top.get("factor_family"),
                min_all_count=min_count,
            ),
            "layer_weights": _count_csv_values(top.get("layer")),
            "window_weights": _count_csv_values(top.get("windows")),
            "group_weights": _count_csv_values(top.get("groups")),
            "negative_field_weights": _compute_frequency_ratio(
                work.get("fields"), negative.get("fields"), min_all_count=min_count
            ),
            "negative_operator_weights": _compute_frequency_ratio(
                work.get("operators"),
                negative.get("operators"),
                min_all_count=min_count,
            ),
            "negative_family_weights": _count_csv_values(negative.get("family")),
            "negative_factor_family_weights": _compute_frequency_ratio(
                work.get("factor_family"),
                negative.get("factor_family"),
                min_all_count=min_count,
            ),
            "negative_layer_weights": _count_csv_values(negative.get("layer")),
            "negative_window_weights": _count_csv_values(negative.get("windows")),
            "negative_group_weights": _count_csv_values(negative.get("groups")),
            "fragment_weights": _count_csv_values(top.get("fragment_hash")),
            "parent_weights": _count_csv_values(top.get("parent_hash")),
            "mutation_type_weights": _count_csv_values(top.get("mutation_type")),
            "negative_fragment_weights": _count_csv_values(negative.get("fragment_hash")),
            "negative_parent_weights": _count_csv_values(negative.get("parent_hash")),
            "negative_mutation_type_weights": _count_csv_values(negative.get("mutation_type")),
        }


def _count_csv_values(series: pd.Series | None) -> dict[str, float]:
    if series is None:
        return {}
    counts: dict[str, float] = {}
    for value in series.dropna().astype(str):
        for item in [x.strip() for x in value.split(",") if x.strip()]:
            counts[item] = counts.get(item, 0.0) + 1.0
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def _compute_frequency_ratio(
    all_series: pd.Series | None,
    top_series: pd.Series | None,
    min_all_count: int = 5,
) -> dict[str, float]:
    """计算 Top 中出现率 / 全部出现率，消除流行度偏差。

    返回值是比值（非概率，不归一化），直接作为相对权重使用。
    下游 candidate_ranker._weight_sum 对权重求和，不假设归一化。

    比值 > 1: 此值在高分表达式中出现比例高于总体 → 正向信号
    比值 = 1: 中性
    比值 < 1: 此值在高分表达式中被抑制 → 负向信号

    当某字段在全部候选中出现次数 < min_all_count 时，回退到简单频率。
    """
    if all_series is None or top_series is None:
        return _count_csv_values(top_series)

    all_counts = _count_csv_values(all_series)
    top_counts = _count_csv_values(top_series)

    all_total = sum(all_counts.values())
    top_total = sum(top_counts.values())

    if all_total <= 0 or top_total <= 0:
        return {}

    weights: dict[str, float] = {}
    for field, top_count in top_counts.items():
        all_count = all_counts.get(field, 0)
        if all_count < min_all_count:
            # 样本太少，回退到简单频率
            weights[field] = top_count / top_total
            continue
        top_rate = top_count / top_total
        all_rate = all_count / all_total
        weights[field] = top_rate / all_rate

    return weights


def _select_feedback_score_column(df: pd.DataFrame) -> str:
    # 优先使用 effectiveness_score（0-100 分制，语义最清晰）
    for col in [
        "effectiveness_score",
        "feedback_score_net",
        "score_total_net",
        "train_score_total_net",
        "feedback_score",
        "score_total_gross",
        "train_score_total",
        "train_score",
        "score_total",
        "scoreboard_score",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col
    return ""


def _score_basis(column: str) -> str:
    text = str(column or "").lower()
    if not text:
        return "none"
    if text.endswith("_net") or "_net_" in text:
        return "net"
    if text.endswith("_gross") or "_gross_" in text:
        return "gross"
    return "fallback"


# ---------------------------------------------------------------------------
# 子结构级别反馈 (T4)
# ---------------------------------------------------------------------------

# 时序算子: 最后一个参数是窗口参数
_TS_OPERATORS = frozenset(
    {
        "ts_rank",
        "ts_zscore",
        "ts_mean",
        "ts_std_dev",
        "ts_delta",
        "ts_min",
        "ts_max",
        "ts_median",
        "ts_av_diff",
        "ts_count_nans",
        "ts_decay_linear",
        "ts_ir",
        "ts_arg_max",
        "ts_arg_min",
        "ts_corr",
        "ts_covariance",
        "ts_backfill",
        "ts_sum",
    }
)


def _node_to_expression(node: ExpressionNode) -> str:
    """将 AST 节点序列化为表达式字符串。"""
    if isinstance(node, LiteralNode):
        val = node.value
        if isinstance(val, str) and val == "*":
            return "*"
        return repr(val)
    if isinstance(node, FieldNode):
        return str(node.name)
    if isinstance(node, FunctionCallNode):
        args = [_node_to_expression(arg) for arg in node.args]
        args.extend(f"{k}={_node_to_expression(v)}" for k, v in sorted(node.named_args.items()))
        return f"{node.name}({', '.join(args)})"
    return ""


def _normalize_windows(node: ExpressionNode) -> ExpressionNode:
    """将时序算子的窗口参数替换为通配符 '*'。"""
    if isinstance(node, FunctionCallNode):
        new_args = list(node.args)
        if node.name in _TS_OPERATORS and new_args:
            last = new_args[-1]
            if isinstance(last, LiteralNode) and isinstance(last.value, (int, float)):
                new_args[-1] = LiteralNode(value="*")
        new_args = [_normalize_windows(arg) for arg in new_args]
        return FunctionCallNode(name=node.name, args=tuple(new_args), named_args=node.named_args)
    if isinstance(node, FieldNode):
        return node
    if isinstance(node, LiteralNode):
        return node
    return node


def _iter_function_calls(node: ExpressionNode) -> list[ExpressionNode]:
    """递归收集 AST 中所有函数调用节点。"""
    result: list[ExpressionNode] = []
    if isinstance(node, FunctionCallNode):
        result.append(node)
        for arg in node.args:
            result.extend(_iter_function_calls(arg))
    return result


def _extract_substructures(expression: str) -> list[str]:
    """提取表达式中的子结构 (忽略窗口参数)。

    示例:
        ts_mean(volume, 10)  → ["ts_mean(volume, *)"]
        rank(ts_mean(volume, 10))  → ["rank(ts_mean(volume, *)), ts_mean(volume, *)"]
    """
    text = str(expression or "").strip()
    if not text:
        return []
    try:
        node = parse_expression(text)
    except (ExpressionParseError, Exception):
        return []
    substructures: list[str] = []
    for func_node in _iter_function_calls(node):
        normalized = _normalize_windows(func_node)
        substructures.append(_node_to_expression(normalized))
    return substructures


def _compute_substructure_frequency_ratio(
    all_expressions: pd.Series | None,
    top_expressions: pd.Series | None,
    min_all_count: int = 5,
) -> dict[str, float]:
    """计算子结构在 Top 中出现率 / 全部出现率。"""
    if all_expressions is None or top_expressions is None:
        return {}

    all_counts: dict[str, float] = {}
    for expr in all_expressions.dropna().astype(str):
        for sub in set(_extract_substructures(expr)):
            all_counts[sub] = all_counts.get(sub, 0.0) + 1.0

    top_counts: dict[str, float] = {}
    for expr in top_expressions.dropna().astype(str):
        for sub in set(_extract_substructures(expr)):
            top_counts[sub] = top_counts.get(sub, 0.0) + 1.0

    all_total = sum(all_counts.values())
    top_total = sum(top_counts.values())
    if all_total <= 0 or top_total <= 0:
        return {}

    weights: dict[str, float] = {}
    for sub, top_count in top_counts.items():
        all_count = all_counts.get(sub, 0)
        if all_count < min_all_count:
            weights[sub] = top_count / top_total
            continue
        top_rate = top_count / top_total
        all_rate = all_count / all_total
        weights[sub] = top_rate / all_rate

    return weights


def _empty_hints(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "field_weights": {},
        "operator_weights": {},
        "substructure_weights": {},
        "family_weights": {},
        "factor_family_weights": {},
        "layer_weights": {},
        "window_weights": {},
        "group_weights": {},
        "negative_field_weights": {},
        "negative_operator_weights": {},
        "negative_family_weights": {},
        "negative_factor_family_weights": {},
        "negative_layer_weights": {},
        "negative_window_weights": {},
        "negative_group_weights": {},
        "fragment_weights": {},
        "parent_weights": {},
        "mutation_type_weights": {},
        "negative_fragment_weights": {},
        "negative_parent_weights": {},
        "negative_mutation_type_weights": {},
    }
