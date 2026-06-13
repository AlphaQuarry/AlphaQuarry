from __future__ import annotations

import pandas as pd

from alpha_mining.mining.expression_canonicalizer import canonicalize_expression
from alpha_mining.mining.fragment_mutation import (
    MutationConfig,
    generate_mutation_candidates,
)


def _fragments_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fragment_hash": "frag_1",
                "fragment_expression": "ts_rank(close, 22)",
                "source_expression": "ts_rank(close, 22)",
                "source_alpha_hash": "parent_1",
            },
            {
                "fragment_hash": "frag_2",
                "fragment_expression": "group_rank(amount, industry)",
                "source_expression": "group_rank(amount, industry)",
                "source_alpha_hash": "parent_2",
            },
        ]
    )


def test_fragment_mutation_generates_deduped_candidates_and_rejects_bad_patterns() -> None:
    existing = {canonicalize_expression("rank(ts_rank(close, 22))").canonical_hash}
    candidates = generate_mutation_candidates(
        fragments_df=_fragments_df(),
        field_roles={
            "close": "price",
            "open": "price",
            "amount": "liquidity",
            "volume": "liquidity",
        },
        group_fields=["industry", "sector", "subindustry"],
        existing_hashes=existing,
        config=MutationConfig(
            windows=(5, 10, 22, 66),
            max_mutations=24,
            max_children_per_parent=3,
            enable_stateful=False,
            random_seed=11,
        ),
    )
    assert candidates
    canonical_hashes = [str(x.get("canonical_hash", "")) for x in candidates]
    assert len(canonical_hashes) == len(set(canonical_hashes))
    assert all("rank(rank(" not in str(x.get("expression", "")).replace(" ", "") for x in candidates)
    assert all("zscore(zscore(" not in str(x.get("expression", "")).replace(" ", "") for x in candidates)
    assert any("ts_rank(close,5)" in str(x.get("canonical_expression", "")) for x in candidates)
    assert all("trade_when_hold(" not in str(x.get("expression", "")) for x in candidates)


def test_fragment_mutation_stateful_budget_cap_is_enforced() -> None:
    candidates = generate_mutation_candidates(
        fragments_df=_fragments_df(),
        field_roles={
            "close": "price",
            "open": "price",
            "amount": "liquidity",
            "volume": "liquidity",
        },
        group_fields=["industry", "sector", "subindustry"],
        existing_hashes=set(),
        config=MutationConfig(
            windows=(5, 10, 22, 66),
            max_mutations=12,
            max_children_per_parent=4,
            enable_stateful=True,
            stateful_ratio_cap=0.10,
            random_seed=13,
        ),
    )
    stateful = [x for x in candidates if str(x.get("mutation_type", "")) == "stateful_wrapper_add"]
    assert len(stateful) <= 1
