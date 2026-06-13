from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from alpha_mining.workflow.closed_loop import ClosedLoopConfig
from alpha_mining.workflow.factor_library import FactorLibraryConfig
from dashboard.api.closed_loop_params import SAFE_SOURCE_CHUNK_HARD_LIMIT_MB, closed_loop_param_schema


def _load_run_closed_loop_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_closed_loop.py"
    spec = importlib.util.spec_from_file_location("run_closed_loop_cli_contract", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_closed_loop_key_defaults_do_not_drift_between_cli_config_and_dashboard_schema() -> None:
    cli = _load_run_closed_loop_module()
    args = cli.build_arg_parser().parse_args([])
    cfg = ClosedLoopConfig()
    schema = closed_loop_param_schema()
    params = {param["name"]: param for group in schema["groups"] for param in group["params"]}

    assert args.request_new == cfg.request_new_alphas == params["request_new"]["default"] == 5
    assert args.batch_size == cfg.batch_size == params["batch_size"]["default"] == 5
    assert args.max_eval == cfg.max_eval_expressions == params["max_eval"]["default"] == 80
    assert args.iterations == cfg.max_iterations == params["iterations"]["default"] == 1
    assert args.source_chunk_loading is False
    assert cfg.enable_source_chunk_loading is True
    assert params["source_chunk_loading"]["default"] is True
    assert cfg.source_chunk_mem_hard_limit_mb == 0.0
    assert params["source_chunk_mem_hard_limit_mb"]["default"] == SAFE_SOURCE_CHUNK_HARD_LIMIT_MB
    assert schema["safe_defaults"]["source_chunk_mem_hard_limit_mb"] == SAFE_SOURCE_CHUNK_HARD_LIMIT_MB
    assert cfg.candidate_artifact_retention_enabled is True
    assert params["candidate_artifact_retention_enabled"]["default"] is True
    assert cfg.analysis_artifact_retention_enabled is False
    assert params["analysis_artifact_retention_enabled"]["default"] is False
    assert cfg.run_health_retention_enabled is True
    assert params["run_health_retention_enabled"]["default"] is True


def test_closed_loop_presets_and_example_yaml_match_safe_resource_contract() -> None:
    schema = closed_loop_param_schema()
    presets = {preset["id"]: preset for preset in schema["presets"]}
    example = yaml.safe_load(Path("configs/closed_loop.example.yaml").read_text(encoding="utf-8"))
    cfg = ClosedLoopConfig()
    lib = FactorLibraryConfig()

    assert presets["smoke"]["params"] == {"request_new": 5, "batch_size": 5, "max_eval": 80, "iterations": 1}
    assert presets["balanced"]["params"] == {"request_new": 20, "batch_size": 10, "max_eval": 500, "iterations": 2}
    assert presets["deep"]["params"] == {"request_new": 50, "batch_size": 10, "max_eval": 2000, "iterations": 3}
    assert example["source_chunk_mem_hard_limit_mb"] == SAFE_SOURCE_CHUNK_HARD_LIMIT_MB
    assert example["candidate_artifact_retention_enabled"] is True
    assert example["analysis_artifact_retention_enabled"] is False
    assert example["run_health_retention_enabled"] is True
    assert cfg.factor_library_min_score == lib.min_score == 60.0
    assert cfg.factor_library_staging_min_score == lib.staging_min_score == 50.0
    assert cfg.factor_library_max_signal_corr == lib.max_signal_corr == 0.80
    assert cfg.factor_library_max_ic_corr == lib.max_ic_corr == 0.80
    assert cfg.factor_library_max_pnl_corr == lib.max_pnl_corr == 0.80
