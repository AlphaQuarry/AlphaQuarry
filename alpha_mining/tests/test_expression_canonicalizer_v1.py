from __future__ import annotations

from alpha_mining.mining.candidate_prefilter import CandidatePrefilter
from alpha_mining.mining.expression_canonicalizer import canonicalize_expression


def test_canonicalizer_normalizes_aliases_commutative_ops_and_constants() -> None:
    left = canonicalize_expression("add(close, volume)")
    right = canonicalize_expression("add(volume, close)")
    assert left.canonical_hash == right.canonical_hash
    assert left.canonical_expression == "add(close,volume)"

    assert canonicalize_expression("subtract(close, 0)").canonical_expression == "close"
    assert canonicalize_expression("multiply(close, 1)").canonical_expression == "close"
    assert canonicalize_expression("reverse(reverse(close))").canonical_expression == "close"
    assert canonicalize_expression("divide(close, volume)").canonical_expression == "div(close,volume)"
    assert (
        canonicalize_expression("cs_quantile(close, 'gaussian', 1.0)").canonical_expression
        == "quantile(close,'gaussian',1)"
    )


def test_prefilter_rejects_bad_canonical_patterns_and_dedupes_by_canonical_hash() -> None:
    field_kinds = {"close": "scalar", "volume": "scalar", "industry": "group"}
    existing = {canonicalize_expression("add(close, volume)").canonical_hash}
    prefilter = CandidatePrefilter(field_kinds=field_kinds, existing_hashes=existing, max_depth=6)

    assert prefilter.check("add(volume, close)").reject_reason == "duplicate_canonical_hash"
    assert prefilter.check("div(close, close)").reject_reason == "self_division"
    assert prefilter.check("rank(rank(close))").reject_reason == "nested_idempotent:rank"
    assert prefilter.check("zscore(zscore(close))").reject_reason == "nested_idempotent:zscore"
    assert prefilter.check("normalize(normalize(close))").reject_reason == "nested_idempotent:normalize"

    result = CandidatePrefilter(field_kinds=field_kinds, max_depth=6).check("multiply(close, 1)")
    assert result.passed
    assert result.canonical_expression == "close"
    assert result.canonical_hash


def test_canonicalizer_sorts_commutative_min_max() -> None:
    left = canonicalize_expression("max(close, volume)")
    right = canonicalize_expression("max(volume, close)")
    assert left.canonical_expression == "max(close,volume)"
    assert left.canonical_hash == right.canonical_hash

    left_min = canonicalize_expression("min(close, volume)")
    right_min = canonicalize_expression("min(volume, close)")
    assert left_min.canonical_expression == "min(close,volume)"
    assert left_min.canonical_hash == right_min.canonical_hash
