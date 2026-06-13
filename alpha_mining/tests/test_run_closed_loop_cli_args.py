from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_run_closed_loop_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_closed_loop.py"
    spec = importlib.util.spec_from_file_location("run_closed_loop_cli_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_layer_budget_parser_accepts_strict_json() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._parse_layer_budget_arg('{"L0":8,"L1":120}') == {"L0": 8, "L1": 120}


def test_run_closed_loop_exposes_reusable_arg_parser() -> None:
    cli = _load_run_closed_loop_module()

    parser = cli.build_arg_parser()
    args = parser.parse_args([])

    assert args.source_backend == "duckdb"
    assert args.search_mode == "layered_v2"
    assert args.max_eval == 80
    assert args.source_chunk_mem_hard_limit_mb == 0.0


def test_layer_budget_parser_accepts_powershell_quote_stripped_keys() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._parse_layer_budget_arg("{L0:8,L1:120,L2:160,L3:100,L4:80}") == {
        "L0": 8,
        "L1": 120,
        "L2": 160,
        "L3": 100,
        "L4": 80,
    }


def test_mutation_budget_ratio_is_normalized_to_unit_interval() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._normalized_mutation_budget_ratio(0.20) == 0.20
    assert cli._normalized_mutation_budget_ratio(2.0) == 1.0
    assert cli._normalized_mutation_budget_ratio(-1.0) == 0.0
    assert cli._normalized_mutation_budget_ratio(0.08) == 0.08


def test_unit_ratio_parser_for_mutation_reserve() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._normalized_unit_ratio(0.25) == 0.25
    assert cli._normalized_unit_ratio(9.0) == 1.0
    assert cli._normalized_unit_ratio(-1.0) == 0.0


def test_positive_int_parser_for_mutation_limits() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._positive_int(3, 5) == 3
    assert cli._positive_int(0, 5) == 1
    assert cli._positive_int("bad", 5) == 5


def test_closed_loop_config_visualization_png_default_off() -> None:
    from alpha_mining.workflow.closed_loop import ClosedLoopConfig

    assert ClosedLoopConfig().include_visualization_png is False


def test_closed_loop_defaults_to_layered_v2_with_132_window() -> None:
    from alpha_mining.mining.explore import DeepExploreConfig
    from alpha_mining.mining.expression_layers import LayeredBuilderConfig
    from alpha_mining.workflow.closed_loop import ClosedLoopConfig

    cli = _load_run_closed_loop_module()

    assert ClosedLoopConfig().search_mode == "layered_v2"
    assert 132 in DeepExploreConfig().windows
    assert cli.DEFAULT_DEEP_WINDOWS == (5, 10, 22, 66, 132)
    assert ClosedLoopConfig().layer_bucket_l1_max_total == 24
    assert ClosedLoopConfig().layer_bucket_l2_max_total == 20
    assert ClosedLoopConfig().layer_enable_recipe_lite is True
    assert ClosedLoopConfig().field_profile_lite_enabled is True
    assert ClosedLoopConfig().feedback_policy_lite_enabled is True
    assert LayeredBuilderConfig().layer_enable_recipe_lite is True


def test_closed_loop_round34_defaults_are_conservative() -> None:
    from alpha_mining.workflow.closed_loop import ClosedLoopConfig

    cfg = ClosedLoopConfig()

    assert cfg.layer_recipe_max_total == 80
    assert cfg.layer_recipe_max_per_family == 16
    assert cfg.layer_role_pair_max_total == 80
    assert cfg.layer_cross_family_pair_ratio == 0.15
    assert cfg.field_profile_lite_min_coverage == 0.20
    assert cfg.field_profile_lite_min_finite_rate == 0.80
    assert cfg.field_profile_lite_top_fields_per_family == 50
    assert cfg.bucket_quality_lite_enabled is True
    assert cfg.bucket_quality_max_evaluations == 80
    assert cfg.bucket_quality_min_coverage == 0.50
    assert cfg.bucket_quality_min_median_group_size == 5
    assert cfg.bucket_quality_min_group_count == 3
    assert cfg.sample_prefilter_stratified is True
    assert cfg.bucket_quality_reject_low_quality_composite is True
    assert cfg.generation_diagnostics_enabled is True
    assert cfg.candidate_artifact_retention_enabled is True
    assert cfg.candidate_artifact_retention_max_batches == 200
    assert cfg.candidate_artifact_retention_days == 30
    assert cfg.analysis_artifact_retention_enabled is False
    assert cfg.analysis_artifact_retention_max_runs == 120
    assert cfg.analysis_artifact_retention_days == 90
    assert cfg.run_health_retention_enabled is True
    assert cfg.run_health_retention_max_lines == 5000
    assert cfg.run_health_retention_days == 90
    assert cfg.registry_health_check is False
    assert cfg.source_chunk_mem_hard_limit_mb == 0.0
    assert cfg.panel_cache_max_size == 64


def test_layer_selection_min_ratio_parser() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._parse_layer_min_ratio_arg('{"L3":0.2,"L4":0.15}') == {
        "L3": 0.2,
        "L4": 0.15,
    }
    assert cli._parse_layer_min_ratio_arg("{L0:0.03,L1:0.25}") == {
        "L0": 0.03,
        "L1": 0.25,
    }


def test_ratio_json_parser_normalizes_layer_and_structure_keys() -> None:
    cli = _load_run_closed_loop_module()

    assert cli._parse_ratio_json_arg("{L0:0.08}", "--layer-selection-max-ratio-json") == {"L0": 0.08}
    assert cli._parse_ratio_json_arg(
        "{bucket:0.1,gate:0.2}",
        "--structure-selection-min-ratio-json",
        normalize_layer_keys=False,
    ) == {
        "bucket": 0.1,
        "gate": 0.2,
    }


def test_closed_loop_config_tradability_default_on_with_reverse_flag_available() -> None:
    from alpha_mining.workflow.closed_loop import ClosedLoopConfig

    cli = _load_run_closed_loop_module()

    assert ClosedLoopConfig().apply_tradability_constraints is True
    assert cli._default_apply_tradability_constraints() is True


def test_benchmark_binding_maps_index_universe_to_matching_benchmark() -> None:
    from alpha_mining.workflow.benchmark_binding import resolve_benchmark_binding

    binding = resolve_benchmark_binding(universe_name="csi500", explicit_code="")

    assert binding.code == "000905.SH"
    assert binding.source == "universe"


def test_factor_library_defaults_are_strict_and_disabled() -> None:
    from alpha_mining.workflow.closed_loop import ClosedLoopConfig
    from alpha_mining.workflow.factor_library import FactorLibraryConfig

    cfg = ClosedLoopConfig()
    lib = FactorLibraryConfig()
    assert cfg.factor_library_enabled is False
    assert cfg.factor_library_min_score == 60.0
    assert cfg.factor_library_max_signal_corr == 0.80
    assert cfg.factor_library_max_ic_corr == 0.80
    assert cfg.factor_library_max_pnl_corr == 0.80
    assert lib.min_score == 60.0
    assert lib.max_signal_corr == 0.80
    assert lib.max_ic_corr == 0.80
    assert lib.max_pnl_corr == 0.80


def test_neutralization_mode_normalization() -> None:
    from alpha_mining.simulation.neutralization import normalize_neutralization_mode

    assert normalize_neutralization_mode(None) == "NONE"
    assert normalize_neutralization_mode("industry") == "INDUSTRY"
    assert normalize_neutralization_mode("GROUP:subindustry") == "SUBINDUSTRY"


def test_neutralization_mode_rejects_unsupported_values() -> None:
    from alpha_mining.simulation.neutralization import normalize_neutralization_mode

    try:
        normalize_neutralization_mode("STATISTICAL")
    except ValueError as exc:
        assert "Unsupported neutralization mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_duckdb_runtime_settings_builds_default_temp_dir() -> None:
    cli = _load_run_closed_loop_module()
    args = Namespace(
        duckdb_memory_limit="6GB",
        duckdb_threads=4,
        duckdb_temp_directory="",
        duckdb_max_temp_directory_size="80GB",
    )
    settings = cli._build_duckdb_runtime_settings(args=args, duckdb_path="D:/project_quant/data/duckdb/market.duckdb")
    assert settings["memory_limit"] == "6GB"
    assert settings["threads"] == 4
    assert settings["temp_directory"].endswith("market.duckdb.tmp")
    assert settings["max_temp_directory_size"] == "80GB"


def test_safe_cleanup_duckdb_temp_dir_only_cleans_default(tmp_path: Path) -> None:
    cli = _load_run_closed_loop_module()
    db_path = tmp_path / "market.duckdb"
    db_path.write_bytes(b"")
    temp_dir = Path(f"{db_path.as_posix()}.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = temp_dir / "duckdb_temp_storage_DEFAULT-0.tmp"
    tmp_file.write_bytes(b"abc")

    cleaned = cli._safe_cleanup_duckdb_temp_dir(str(db_path), str(temp_dir))
    assert cleaned["deleted_files"] >= 1
    assert cleaned["deleted_bytes"] >= 3
    assert not temp_dir.exists()

    custom_dir = tmp_path / "custom_tmp_dir"
    custom_dir.mkdir(parents=True, exist_ok=True)
    (custom_dir / "x.tmp").write_bytes(b"abc")
    skipped = cli._safe_cleanup_duckdb_temp_dir(str(db_path), str(custom_dir))
    assert skipped["skipped"] == "non_default_temp_directory"
    assert custom_dir.exists()


def test_safe_cleanup_duckdb_temp_dir_allows_nested_default_with_flag(
    tmp_path: Path,
) -> None:
    cli = _load_run_closed_loop_module()
    db_path = tmp_path / "market.duckdb"
    db_path.write_bytes(b"")
    base_temp = Path(f"{db_path.as_posix()}.tmp")
    nested = base_temp / "run_u1"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "spill.tmp").write_bytes(b"abcdef")

    skipped = cli._safe_cleanup_duckdb_temp_dir(
        str(db_path),
        str(nested),
        allow_nested_default=False,
    )
    assert skipped["skipped"] == "non_default_temp_directory"
    assert nested.exists()

    cleaned = cli._safe_cleanup_duckdb_temp_dir(
        str(db_path),
        str(nested),
        allow_nested_default=True,
    )
    assert cleaned["deleted_files"] >= 1
    assert cleaned["deleted_bytes"] >= 6
    assert not nested.exists()
