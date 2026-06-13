from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    expression: str
    normalized_expression: str
    expression_hash: str
    structural_hash: str
    source: str
    generation_mode: str
    family: str
    original_expression: str = ""
    simplified_expression: str = ""
    factor_family: str = ""
    factor_family_mix_json: str = "{}"
    primary_factor_family: str = ""
    layer: str = ""
    layer_family: str = ""
    parent_expression: str = ""
    parent_hash: str = ""
    mutation_type: str = ""
    fragment_hash: str = ""
    feedback_source: str = ""
    builder_source: str = ""
    layer_order: int = 0
    canonical_expression: str = ""
    canonical_hash: str = ""
    lint_passed: bool = True
    lint_reject_reason: str = ""
    lint_status: str = "passed"
    lint_warning_reason: str = ""
    template_id: str = ""
    fields: str = ""
    field_roles: str = ""
    groups: str = ""
    operators: str = ""
    operator_count: int = 0
    field_count: int = 0
    depth: int = 0
    windows: str = ""
    pair_key: str = ""
    random_seed: int = 0
    generation_iteration: int = 0
    field_profile_score: float = 0.0
    recipe_score: float = 0.0
    role_pair_score: float = 0.0
    bucket_quality_score: float = 0.0
    gate_quality_score: float = 0.0
    sample_quality_score: float = 0.0
    sample_coverage: float = 0.0
    sample_inf_ratio: float = 0.0
    sample_extreme_ratio: float = 0.0
    sample_unique_count: int = 0
    bucket_sample_quality_score: float = 0.0
    bucket_sample_status: str = ""
    bucket_sample_reject_reason: str = ""
    bucket_sample_coverage: float = 0.0
    bucket_sample_group_count_median: float = 0.0
    bucket_sample_group_size_median: float = 0.0
    bucket_sample_group_size_min: float = 0.0
    cost_score: float = 0.0
    complexity_score: float = 0.0
    candidate_score: float = 0.0
    feedback_score: float = 0.0
    fragment_score: float = 0.0
    parent_score: float = 0.0
    mutation_type_score: float = 0.0
    novelty_score: float = 0.0
    sample_status: str = ""
    sample_reject_reason: str = ""
    prefilter_status: str = "pending"
    reject_stage: str = ""
    reject_reason: str = ""
    metadata_json: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
