from __future__ import annotations

import random
from dataclasses import dataclass, field
from .field_preprocessing import FieldExpressionFactory, FieldPreprocessConfig


@dataclass(frozen=True)
class DeepExploreConfig:
    """Config for operator-driven deep exploration candidate generation."""

    windows: tuple[int, ...] = (5, 10, 22, 66, 132)
    max_depth: int = 2
    max_candidates: int = 300
    max_inputs_per_operator: int = 24
    max_binary_pairs: int = 120
    include_group_ops: bool = True
    include_binary_ops: bool = True
    enable_stateful_phase2_ops: bool = False
    field_preprocess_config: FieldPreprocessConfig = field(default_factory=FieldPreprocessConfig)
    random_seed: int = 42


@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_kind: str = "scalar"  # scalar / vector / group / mask
    categories: tuple[str, ...] = ()
    factor_family: str = ""


@dataclass(frozen=True)
class OperatorSignature:
    name: str
    input_types: tuple[str, ...]
    output_type: str = "scalar"


class RandomExpressionGenerator:
    """
    Signature-aware random expression generator.

    The generator stays compatible with current operator-only style while
    reducing invalid combinations through lightweight type constraints.
    """

    def __init__(
        self,
        field_specs: list[FieldSpec] | None = None,
        group_fields: list[str] | None = None,
        windows: list[int] | None = None,
        max_depth: int = 2,
        random_seed: int = 42,
        enable_stateful_phase2_ops: bool = False,
        scalar_fields: list[str] | None = None,
        vector_fields: list[str] | None = None,
        field_preprocess_config: FieldPreprocessConfig | None = None,
    ) -> None:
        self.rng = random.Random(int(random_seed))
        if field_specs is None:
            specs = [FieldSpec(name=str(x), field_kind="scalar") for x in (scalar_fields or [])]
            specs.extend([FieldSpec(name=str(x), field_kind="vector") for x in (vector_fields or [])])
            field_specs = specs
        self.field_specs = list(field_specs)
        self.group_fields = list(group_fields or [])
        self.windows = list(windows or [5, 10, 22, 66, 132])
        self.max_depth = max(1, int(max_depth))
        self.enable_stateful_phase2_ops = bool(enable_stateful_phase2_ops)
        self.field_expression_factory = FieldExpressionFactory(field_preprocess_config)
        self.scalar_fields = [f.name for f in self.field_specs if f.field_kind == "scalar"]
        self.vector_fields = [f.name for f in self.field_specs if f.field_kind == "vector"]

        self.unary_ops = [
            OperatorSignature("rank", ("scalar",)),
            OperatorSignature("zscore", ("scalar",)),
            OperatorSignature("normalize", ("scalar",)),
            OperatorSignature("reverse", ("scalar",)),
            OperatorSignature("quantile", ("scalar",)),
            OperatorSignature("truncate", ("scalar", "literal")),
            OperatorSignature("left_tail", ("scalar", "literal")),
            OperatorSignature("right_tail", ("scalar", "literal")),
        ]
        self.ts_ops = [
            OperatorSignature("ts_rank", ("scalar", "window")),
            OperatorSignature("ts_zscore", ("scalar", "window")),
            OperatorSignature("ts_mean", ("scalar", "window")),
            OperatorSignature("ts_std_dev", ("scalar", "window")),
            OperatorSignature("ts_delta", ("scalar", "window")),
            OperatorSignature("ts_delay", ("scalar", "window")),
            OperatorSignature("ts_min", ("scalar", "window")),
            OperatorSignature("ts_max", ("scalar", "window")),
            OperatorSignature("ts_median", ("scalar", "window")),
            OperatorSignature("ts_av_diff", ("scalar", "window")),
            OperatorSignature("ts_count_nans", ("scalar", "window")),
        ]
        self.ts_pair_ops = [
            OperatorSignature("ts_corr", ("scalar", "scalar", "window")),
            OperatorSignature("ts_covariance", ("scalar", "scalar", "window")),
        ]
        self.group_ops = [
            OperatorSignature("group_rank", ("scalar", "group")),
            OperatorSignature("group_zscore", ("scalar", "group")),
            OperatorSignature("group_neutralize", ("scalar", "group")),
            OperatorSignature("group_median", ("scalar", "group")),
            OperatorSignature("group_scale", ("scalar", "group")),
        ]
        self.binary_ops = ["add", "sub", "mul", "div"]

    def generate(self, max_candidates: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        attempts = max(64, int(max_candidates) * 8)
        for _ in range(attempts):
            depth = self.rng.randint(1, self.max_depth)
            expr = self._gen_scalar_expr(depth)
            if not expr:
                continue
            key = "".join(expr.split())
            if key in seen:
                continue
            seen.add(key)
            out.append(expr)
            if len(out) >= int(max_candidates):
                break
        return out

    def _gen_scalar_expr(self, depth: int) -> str:
        if depth <= 0:
            return self._sample_base_expr()

        choices = ["base", "unary", "ts", "ts_pair", "binary"]
        if self.group_fields:
            choices.append("group")
        if self.enable_stateful_phase2_ops:
            choices.append("stateful")
        op = self.rng.choice(choices)
        if op == "base":
            return self._sample_base_expr()
        if op == "unary":
            seed = self._gen_scalar_expr(depth - 1)
            sig = self.rng.choice(self.unary_ops)
            if sig.name == "truncate":
                bound = self.rng.choice([0.01, 0.02, 0.05])
                return f"{sig.name}({seed}, {bound})"
            if sig.name == "left_tail":
                return f"{sig.name}({seed}, 0.0)"
            if sig.name == "right_tail":
                return f"{sig.name}({seed}, 0.0)"
            if sig.name == "quantile":
                mode = self.rng.choice(["'gaussian'", "'uniform'"])
                return f"{sig.name}({seed}, {mode}, 1.0)"
            return f"{sig.name}({seed})"
        if op == "ts":
            seed = self._gen_scalar_expr(depth - 1)
            sig = self.rng.choice(self.ts_ops)
            d = self.rng.choice(self.windows)
            return f"{sig.name}({seed}, {d})"
        if op == "ts_pair":
            left = self._gen_scalar_expr(depth - 1)
            right = self._gen_scalar_expr(depth - 1)
            sig = self.rng.choice(self.ts_pair_ops)
            d = self.rng.choice(self.windows)
            return f"{sig.name}({left}, {right}, {d})"
        if op == "group":
            seed = self._gen_scalar_expr(depth - 1)
            sig = self.rng.choice(self.group_ops)
            g = self.rng.choice(self.group_fields)
            return f"{sig.name}({seed}, {g})"
        if op == "stateful":
            seed = self._gen_scalar_expr(depth - 1)
            d = self.rng.choice(self.windows)
            gate = f"greater(rank({seed}), ts_mean(rank({seed}), {d}))"
            exit_gate = f"less(rank({seed}), ts_mean(rank({seed}), {d}))"
            if self.rng.random() < 0.5:
                return f"hump({seed}, 0.01)"
            return f"trade_when_hold({gate}, {seed}, {exit_gate})"
        # binary
        left = self._gen_scalar_expr(depth - 1)
        right = self._gen_scalar_expr(depth - 1)
        bop = self.rng.choice(self.binary_ops)
        return f"{bop}({left}, {right})"

    def _sample_base_expr(self) -> str:
        options: list[str] = []
        options.extend([self.field_expression_factory.expression_for(x, kind="scalar") for x in self.scalar_fields])
        options.extend([f"vec_avg({x})" for x in self.vector_fields])
        options.extend([f"vec_sum({x})" for x in self.vector_fields])
        options.extend([f"vec_stddev({x})" for x in self.vector_fields])
        if not options:
            return ""
        return self.rng.choice(options)


def build_signature_aware_search_space(
    available_fields: set[str] | list[str],
    available_groups: set[str] | list[str] | None = None,
    config: DeepExploreConfig | None = None,
    field_specs: list[FieldSpec] | None = None,
    excluded_fields: set[str] | list[str] | None = None,
) -> list[tuple[str, str]]:
    cfg = config or DeepExploreConfig()
    excluded = {str(f) for f in (excluded_fields or []) if str(f)}
    fields = sorted({str(f) for f in available_fields if str(f) and str(f) not in excluded})
    groups = sorted({str(g) for g in (available_groups or []) if str(g)})
    if not fields:
        return []

    if field_specs is None:
        specs = [FieldSpec(name=f, field_kind="scalar") for f in fields]
    else:
        spec_map = {str(s.name): s for s in field_specs if str(s.name) in set(fields)}
        specs = [spec_map[f] for f in fields if f in spec_map]
        missing = [f for f in fields if f not in spec_map]
        specs.extend([FieldSpec(name=f, field_kind="scalar") for f in missing])

    windows = _sanitize_windows(cfg.windows)
    gen = RandomExpressionGenerator(
        field_specs=specs,
        group_fields=groups,
        windows=windows,
        max_depth=int(cfg.max_depth),
        random_seed=int(cfg.random_seed),
        enable_stateful_phase2_ops=bool(cfg.enable_stateful_phase2_ops),
        field_preprocess_config=cfg.field_preprocess_config,
    )
    expressions = gen.generate(max_candidates=int(cfg.max_candidates))
    return [("op_signature", expr) for expr in expressions]


def build_operator_search_space(
    available_fields: set[str] | list[str],
    available_groups: set[str] | list[str] | None = None,
    config: DeepExploreConfig | None = None,
    excluded_fields: set[str] | list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Build deeper operator-composed expression candidates beyond template-only mode.

    Parameters:
    - available_fields: scalar/vector field names that can be used as operator inputs.
    - excluded_fields: fields to exclude from operator candidate generation
      (for example, simulation-only mask fields such as "universe").

    Returns list of (source_tag, expression).
    """
    cfg = config or DeepExploreConfig()
    rng = random.Random(int(cfg.random_seed))

    excluded = {str(f) for f in (excluded_fields or []) if str(f)}
    fields = sorted({str(f) for f in available_fields if str(f) and str(f) not in excluded})
    groups = sorted({str(g) for g in (available_groups or []) if str(g)})
    windows = _sanitize_windows(cfg.windows)
    if not fields:
        return []

    generated: list[tuple[str, str]] = []
    expr_map = FieldExpressionFactory(cfg.field_preprocess_config).expression_map(fields)

    # Depth-0: scalar base fields.
    for f in fields:
        generated.append(("op_base", expr_map.get(f, f)))

    # Depth-1: common one-hop transforms.
    depth1_exprs: list[str] = []
    for f in fields:
        f_expr = expr_map.get(f, f)
        depth1_exprs.extend(
            [
                f"rank({f_expr})",
                f"zscore({f_expr})",
                f"normalize({f_expr})",
                f"reverse({f_expr})",
                f"quantile({f_expr}, 'gaussian', 1.0)",
                f"quantile({f_expr}, 'uniform', 1.0)",
                f"truncate({f_expr}, 0.02)",
                f"left_tail({f_expr}, 0.0)",
                f"right_tail({f_expr}, 0.0)",
            ]
        )
        for d in windows:
            depth1_exprs.extend(
                [
                    f"ts_rank({f_expr}, {d})",
                    f"ts_zscore({f_expr}, {d})",
                    f"ts_mean({f_expr}, {d})",
                    f"ts_std_dev({f_expr}, {d})",
                    f"ts_delta({f_expr}, {d})",
                    f"ts_delay({f_expr}, {d})",
                    f"ts_min({f_expr}, {d})",
                    f"ts_max({f_expr}, {d})",
                    f"ts_median({f_expr}, {d})",
                    f"ts_av_diff({f_expr}, {d})",
                    f"ts_count_nans({f_expr}, {d})",
                ]
            )

        # Cross-window momentum/reversion style templates.
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                d_short = windows[i]
                d_long = windows[j]
                depth1_exprs.append(f"rank(ts_mean({f_expr}, {d_short}) - ts_mean({f_expr}, {d_long}))")
                depth1_exprs.append(f"rank(ts_mean({f_expr}, {d_long}) - ts_mean({f_expr}, {d_short}))")

    if cfg.include_group_ops and groups:
        for f in fields:
            f_expr = expr_map.get(f, f)
            for g in groups:
                depth1_exprs.extend(
                    [
                        f"group_rank({f_expr}, {g})",
                        f"group_zscore({f_expr}, {g})",
                        f"group_neutralize({f_expr}, {g})",
                        f"group_median({f_expr}, {g})",
                        f"group_scale({f_expr}, {g})",
                    ]
                )

    for expr in depth1_exprs:
        generated.append(("op_depth1", expr))

    # Depth-2: compose on top of selected depth-1 seeds.
    if int(cfg.max_depth) >= 2:
        seeds_for_depth2 = _sample_keep_order(
            depth1_exprs,
            max_items=max(int(cfg.max_inputs_per_operator), 0),
            rng=rng,
        )
        depth2_exprs: list[str] = []
        short_windows = windows[: min(3, len(windows))]
        for seed in seeds_for_depth2:
            depth2_exprs.extend(
                [
                    f"rank({seed})",
                    f"zscore({seed})",
                    f"normalize({seed})",
                ]
            )
            for d in short_windows:
                depth2_exprs.extend(
                    [
                        f"ts_rank({seed}, {d})",
                        f"ts_zscore({seed}, {d})",
                        f"ts_mean({seed}, {d})",
                        f"ts_delta({seed}, {d})",
                        f"ts_av_diff({seed}, {d})",
                    ]
                )
            if cfg.include_group_ops and groups:
                for g in groups:
                    depth2_exprs.extend(
                        [
                            f"group_rank({seed}, {g})",
                            f"group_zscore({seed}, {g})",
                        ]
                    )
        for expr in depth2_exprs:
            generated.append(("op_depth2", expr))

    # Optional binary composition layer (still parser-portable and bounded).
    if cfg.include_binary_ops:
        left_pool = _sample_keep_order(
            [x for x in depth1_exprs if x.startswith("rank(") or x.startswith("zscore(")],
            max_items=max(int(cfg.max_inputs_per_operator), 0),
            rng=rng,
        )
        right_pool = _sample_keep_order(
            [x for x in depth1_exprs if x.startswith("ts_") or x.startswith("group_")],
            max_items=max(int(cfg.max_inputs_per_operator), 0),
            rng=rng,
        )
        pairs = [(left, r) for left in left_pool for r in right_pool]
        if pairs:
            rng.shuffle(pairs)
            pairs = pairs[: max(int(cfg.max_binary_pairs), 0)]
            for left, right in pairs:
                generated.extend(
                    [
                        ("op_binary", f"add({left}, {right})"),
                        ("op_binary", f"sub({left}, {right})"),
                        ("op_binary", f"mul({left}, {right})"),
                        ("op_binary", f"div({left}, {right})"),
                        ("op_binary", f"ts_covariance({left}, {right}, {windows[0]})"),
                    ]
                )

    if bool(cfg.enable_stateful_phase2_ops):
        stateful_seeds = _sample_keep_order(
            [x for x in depth1_exprs if x.startswith(("rank(", "zscore(", "ts_mean(", "ts_zscore("))],
            max_items=max(int(cfg.max_inputs_per_operator), 0),
            rng=rng,
        )
        d = windows[0]
        for seed in stateful_seeds:
            gate = f"greater(rank({seed}), ts_mean(rank({seed}), {d}))"
            exit_gate = f"less(rank({seed}), ts_mean(rank({seed}), {d}))"
            generated.append(("op_stateful", f"hump({seed}, 0.01)"))
            generated.append(("op_stateful", f"trade_when_hold({gate}, {seed}, {exit_gate})"))

    deduped = _dedupe_pair_by_expr(generated)
    if int(cfg.max_candidates) > 0 and len(deduped) > int(cfg.max_candidates):
        keep = deduped[: min(32, len(deduped))]
        tail = deduped[len(keep) :]
        rng.shuffle(tail)
        keep.extend(tail[: int(cfg.max_candidates) - len(keep)])
        deduped = keep
    return deduped


def _sanitize_windows(values: tuple[int, ...] | list[int]) -> list[int]:
    out = sorted({int(v) for v in values if int(v) > 0})
    return out if out else [5, 10, 22]


def _dedupe_pair_by_expr(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for source, expr in items:
        key = "".join(str(expr).split())
        if key in seen:
            continue
        seen.add(key)
        out.append((source, expr))
    return out


def _sample_keep_order(items: list[str], max_items: int, rng: random.Random) -> list[str]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    idx = sorted(rng.sample(range(len(items)), k=max_items))
    return [items[i] for i in idx]
