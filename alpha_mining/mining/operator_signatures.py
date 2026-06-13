from __future__ import annotations

from dataclasses import dataclass


SCALAR = "scalar"
VECTOR = "vector"
GROUP = "group"
BOOL = "bool"
LITERAL = "literal"
WINDOW = "window"


@dataclass(frozen=True)
class OperatorSignatureSpec:
    name: str
    input_types: tuple[str, ...]
    output_type: str


class OperatorSignatureRegistry:
    def __init__(self) -> None:
        self._items: dict[str, list[OperatorSignatureSpec]] = {}

    def register(self, name: str, input_types: tuple[str, ...], output_type: str) -> None:
        key = str(name)
        spec = OperatorSignatureSpec(key, tuple(input_types), str(output_type))
        specs = self._items.setdefault(key, [])
        if spec not in specs:
            specs.append(spec)

    def get(self, name: str) -> OperatorSignatureSpec | None:
        specs = self._items.get(str(name), [])
        return specs[0] if specs else None

    def get_all(self, name: str) -> tuple[OperatorSignatureSpec, ...]:
        return tuple(self._items.get(str(name), ()))

    def match(self, name: str, arg_types: tuple[str, ...]) -> OperatorSignatureSpec | None:
        for spec in self.get_all(name):
            if len(spec.input_types) != len(arg_types):
                continue
            if all(_type_matches(got=got, want=want) for got, want in zip(arg_types, spec.input_types)):
                return spec
        return None

    def has(self, name: str) -> bool:
        return str(name) in self._items

    def names(self) -> list[str]:
        return sorted(self._items)


def build_default_operator_signature_registry() -> OperatorSignatureRegistry:
    reg = OperatorSignatureRegistry()
    for name in [
        "rank",
        "zscore",
        "normalize",
        "reverse",
        "abs",
        "sign",
        "log",
        "sqrt",
        "scale",
        "quantile",
        "cs_quantile",
        "truncate",
        "left_tail",
        "right_tail",
        "zero_like",
    ]:
        reg.register(name, (SCALAR,), SCALAR)
    reg.register("scale", (SCALAR, LITERAL), SCALAR)
    reg.register("scale", (SCALAR, LITERAL, LITERAL, LITERAL), SCALAR)
    reg.register("quantile", (SCALAR, LITERAL), SCALAR)
    reg.register("quantile", (SCALAR, LITERAL, LITERAL), SCALAR)
    reg.register("cs_quantile", (SCALAR, LITERAL), SCALAR)
    reg.register("cs_quantile", (SCALAR, LITERAL, LITERAL), SCALAR)
    reg.register("truncate", (SCALAR, LITERAL), SCALAR)
    reg.register("left_tail", (SCALAR, LITERAL), SCALAR)
    reg.register("right_tail", (SCALAR, LITERAL), SCALAR)
    for name in ["inverse", "s_log_1p"]:
        reg.register(name, (SCALAR,), SCALAR)
    reg.register("winsorize", (SCALAR, LITERAL), SCALAR)
    for name in ["power", "signed_power"]:
        reg.register(name, (SCALAR, LITERAL), SCALAR)
        reg.register(name, (SCALAR, SCALAR), SCALAR)
    for name in ["max", "min"]:
        reg.register(name, (SCALAR, SCALAR), SCALAR)
    for name in [
        "ts_rank",
        "ts_zscore",
        "ts_mean",
        "ts_std_dev",
        "ts_delta",
        "ts_delay",
        "ts_sum",
        "ts_ir",
        "ts_decay_linear",
        "ts_min",
        "ts_max",
        "ts_median",
        "ts_av_diff",
        "ts_count_nans",
        "ts_backfill",
    ]:
        reg.register(name, (SCALAR, WINDOW), SCALAR)
    for name in ["ts_arg_max", "ts_arg_min", "ts_product"]:
        reg.register(name, (SCALAR, WINDOW), SCALAR)
    reg.register("ts_decay_exp_window", (SCALAR, WINDOW), SCALAR)
    reg.register("ts_decay_exp_window", (SCALAR, WINDOW, LITERAL), SCALAR)
    reg.register("ts_corr", (SCALAR, SCALAR, WINDOW), SCALAR)
    reg.register("ts_covariance", (SCALAR, SCALAR, WINDOW), SCALAR)
    reg.register("ts_regression", (SCALAR, SCALAR, WINDOW), SCALAR)
    reg.register("ts_regression", (SCALAR, SCALAR, WINDOW, LITERAL), SCALAR)
    reg.register("hump", (SCALAR,), SCALAR)
    reg.register("hump", (SCALAR, LITERAL), SCALAR)
    for name in [
        "group_rank",
        "group_zscore",
        "group_neutralize",
        "group_mean",
        "group_normalize",
        "group_sum",
        "group_median",
        "group_scale",
    ]:
        reg.register(name, (SCALAR, GROUP), SCALAR)
    reg.register("regression_neut", (SCALAR, SCALAR), SCALAR)
    reg.register("event_active", (BOOL, SCALAR), SCALAR)
    reg.register("event_decay", (BOOL, SCALAR), SCALAR)
    reg.register("event_decay", (BOOL, SCALAR, LITERAL), SCALAR)
    reg.register("days_from_last_change", (SCALAR,), SCALAR)
    reg.register("densify", (GROUP,), GROUP)
    reg.register("bucket", (SCALAR, LITERAL), GROUP)
    reg.register("group_cartesian_product", (GROUP, GROUP), GROUP)
    for name in ["vec_avg", "vec_sum", "vec_stddev", "vec_max", "vec_min", "vec_count"]:
        reg.register(name, (VECTOR,), SCALAR)
    for name in [
        "greater",
        "less",
        "greater_equal",
        "less_equal",
        "equal",
        "not_equal",
    ]:
        reg.register(name, (SCALAR, SCALAR), BOOL)
        reg.register(name, (SCALAR, LITERAL), BOOL)
        reg.register(name, (LITERAL, SCALAR), BOOL)
    reg.register("is_nan", (SCALAR,), BOOL)
    reg.register("is_not_nan", (SCALAR,), BOOL)
    reg.register("if_else", (BOOL, SCALAR, SCALAR), SCALAR)
    reg.register("trade_when", (BOOL, SCALAR, SCALAR), SCALAR)
    reg.register("trade_when_hold", (BOOL, SCALAR, BOOL), SCALAR)
    for name in ["add", "sub", "mul", "div", "subtract", "multiply", "divide"]:
        reg.register(name, (SCALAR, SCALAR), SCALAR)
    return reg


def _type_matches(got: str, want: str) -> bool:
    if want == WINDOW and got == LITERAL:
        return True
    if want == LITERAL and got == LITERAL:
        return True
    return got == want
