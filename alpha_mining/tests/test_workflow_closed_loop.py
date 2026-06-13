from __future__ import annotations

import shutil
import unittest
import uuid
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_mining.workflow.closed_loop import (
    ClosedLoopConfig,
    _acquire_loop_lock,
    _append_run_health,
    _classify_closed_loop_failure,
    _release_loop_lock,
    prune_analysis_artifacts,
    run_one_loop_iteration,
    validate_universe_registries,
)
from alpha_mining.workflow.lifecycle import load_lifecycle_registry
from alpha_mining.workflow.universe_store import (
    append_universe_expressions,
    load_universe_base_frame,
    load_universe_expression_registry,
)


class TestWorkflowClosedLoop(unittest.TestCase):
    def test_expression_registry_treats_neutralization_as_signal_identity(self) -> None:
        base_dir = Path("data") / f"_closed_loop_signal_hash_{uuid.uuid4().hex}"
        universe = "cn_signal_hash_test"
        try:
            first = append_universe_expressions(
                pd.DataFrame(
                    [
                        {
                            "expression": "rank(close)",
                            "simulation_config_json": '{"neutralization":"NONE","delay":1}',
                        }
                    ]
                ),
                base_dir=str(base_dir),
                universe_name=universe,
            )
            second = append_universe_expressions(
                pd.DataFrame(
                    [
                        {
                            "expression": "rank(close)",
                            "simulation_config_json": '{"neutralization":"INDUSTRY","delay":1}',
                        }
                    ]
                ),
                base_dir=str(base_dir),
                universe_name=universe,
            )
            duplicate = append_universe_expressions(
                pd.DataFrame(
                    [
                        {
                            "expression": "rank(close)",
                            "simulation_config_json": '{"delay":1,"neutralization":"industry"}',
                        }
                    ]
                ),
                base_dir=str(base_dir),
                universe_name=universe,
            )

            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertTrue(duplicate.empty)
            registry = load_universe_expression_registry(base_dir=str(base_dir), universe_name=universe)
            self.assertEqual(set(registry["neutralization"].astype(str)), {"NONE", "INDUSTRY"})
            self.assertEqual(len(set(registry["signal_hash"].astype(str))), 2)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_run_one_iteration_chunked_and_purged(self) -> None:
        rng = np.random.RandomState(7)
        dates = pd.date_range("2024-01-01", periods=24, freq="D")
        codes = ["A", "B", "C", "D", "E"]
        rows: list[dict[str, object]] = []
        for d in dates:
            for c in codes:
                rows.append(
                    {
                        "date": d,
                        "code": c,
                        "pct_chg": float(rng.normal(0.001, 0.02)),
                        "circ_mv": float(rng.uniform(5e8, 3e9)),
                        "close": float(rng.uniform(10, 200)),
                        "volume": float(rng.uniform(1e5, 5e6)),
                        "industry": "I1" if c in {"A", "B", "C"} else "I2",
                        "sector": "S1" if c in {"A", "B"} else "S2",
                        "universe": 1.0,
                    }
                )
        raw_df = pd.DataFrame(rows)

        base_dir = Path("data") / f"_closed_loop_test_{uuid.uuid4().hex}"
        universe = "cn_closed_loop_test"
        try:
            cfg = ClosedLoopConfig(
                universe_name=universe,
                universe_base_dir=str(base_dir),
                batch_size=5,
                request_new_alphas=7,
                max_new_alphas_per_chunk=3,
                compute_chunk_size=3,
                max_eval_expressions=120,
                search_mode="operator_only",
                enable_purge_after_analysis=True,
                analysis_layers=5,
                analysis_include_robustness=False,
            )
            out = run_one_loop_iteration(raw_df=raw_df, config=cfg)
            self.assertEqual(out.get("status"), "ok")
            self.assertIn("artifact_retention_summary", out)
            self.assertIn("run_health_path", out)
            self.assertTrue(Path(str(out["run_health_path"])).exists())
            health_lines = Path(str(out["run_health_path"])).read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(health_lines), 1)
            self.assertEqual(json.loads(health_lines[-1])["status"], "ok")

            alpha_names = [str(x) for x in out.get("alpha_names", [])]
            self.assertGreaterEqual(len(alpha_names), 1)
            self.assertLessEqual(len(alpha_names), 7)

            chunk_results = out.get("chunk_results", [])
            self.assertTrue(isinstance(chunk_results, list))
            for chunk in chunk_results:
                self.assertLessEqual(len(chunk.get("alpha_names", [])), 3)
                meta = chunk.get("analysis_meta", {})
                table_paths = dict(meta.get("table_paths") or {}) if isinstance(meta, dict) else {}
                self.assertNotIn("visualization_manifest", table_paths)
                for key in ["analysis_distribution_histogram", "analysis_ic_decay"]:
                    self.assertIn(key, table_paths)
                    self.assertTrue(Path(table_paths[key]).exists())
                stored_meta = json.loads(
                    (Path(meta["analysis_dir"]) / "analysis_meta.json").read_text(encoding="utf-8")
                )
                self.assertNotIn("visualization_manifest", dict(stored_meta["table_paths"]))

            lifecycle = load_lifecycle_registry(base_dir=str(base_dir), universe_name=universe)
            self.assertTrue(isinstance(lifecycle, pd.DataFrame))
            if not lifecycle.empty:
                life_map = {
                    str(r["alpha_name"]): str(r["status"])
                    for _, r in lifecycle.iterrows()
                    if str(r.get("alpha_name", ""))
                }
                for alpha_name in alpha_names:
                    self.assertEqual(life_map.get(alpha_name), "PURGED")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_atomic_closed_loop_lock_blocks_competing_owner_and_preserves_owner_release(
        self,
    ) -> None:
        base_dir = Path("data") / f"_closed_loop_lock_test_{uuid.uuid4().hex}"
        lock_path = base_dir / ".closed_loop.lock"
        try:
            first = _acquire_loop_lock(
                lock_path=lock_path,
                timeout_seconds=3600,
                universe_name="u1",
                config_hash="cfg1",
            )
            self.assertTrue((lock_path / "owner.json").exists())
            with self.assertRaises(RuntimeError):
                _acquire_loop_lock(
                    lock_path=lock_path,
                    timeout_seconds=3600,
                    universe_name="u1",
                    config_hash="cfg2",
                )

            _release_loop_lock(lock_path=lock_path, owner_id="wrong-owner")
            self.assertTrue(lock_path.exists())
            _release_loop_lock(lock_path=lock_path, owner_id=str(first["owner_id"]))
            self.assertFalse(lock_path.exists())
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_atomic_closed_loop_lock_recovers_stale_and_corrupt_owner(self) -> None:
        base_dir = Path("data") / f"_closed_loop_lock_stale_test_{uuid.uuid4().hex}"
        lock_path = base_dir / ".closed_loop.lock"
        try:
            lock_path.mkdir(parents=True)
            (lock_path / "owner.json").write_text("{bad json", encoding="utf-8")
            owner = _acquire_loop_lock(
                lock_path=lock_path,
                timeout_seconds=0,
                universe_name="u1",
                config_hash="cfg",
            )
            self.assertEqual(owner["universe"], "u1")
            self.assertTrue((lock_path / "owner.json").exists())
            _release_loop_lock(lock_path=lock_path, owner_id=str(owner["owner_id"]))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_closed_loop_failure_classifier_separates_permanent_and_retryable(
        self,
    ) -> None:
        permanent = _classify_closed_loop_failure(
            ValueError("Missing required columns in source view: ['close']"),
            stage="materialize",
        )
        retryable = _classify_closed_loop_failure(PermissionError("file is locked"), stage="materialize")

        self.assertEqual(permanent["status"], "PERMANENT_FAILED")
        self.assertEqual(permanent["failure_kind"], "permanent")
        self.assertEqual(retryable["status"], "FAILED")
        self.assertEqual(retryable["failure_kind"], "retryable")

    def test_analysis_artifact_retention_prunes_old_runs_only(self) -> None:
        base_dir = Path("data") / f"_analysis_retention_test_{uuid.uuid4().hex}"
        universe = "cn_analysis_retention"
        root = base_dir / universe
        try:
            analysis_root = root / "analysis"
            registry = analysis_root / "analysis_registry.csv"
            metrics = analysis_root / "factor_metrics_registry.csv"
            superalpha = root / "superalphas" / "superalpha_keep" / "meta.json"
            catalog = root / "catalog" / "expressions.csv"
            for path in [registry, metrics, superalpha, catalog]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep", encoding="utf-8")
            for idx in range(3):
                run_dir = analysis_root / "period_1" / f"analysis_alpha{idx}_l10_ts1"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "analysis_meta.json").write_text("{}", encoding="utf-8")

            summary = prune_analysis_artifacts(root=root, max_runs=1, retention_days=0)

            remaining = sorted(p.name for p in (analysis_root / "period_1").iterdir() if p.is_dir())
            self.assertEqual(remaining, ["analysis_alpha2_l10_ts1"])
            self.assertTrue(registry.exists())
            self.assertTrue(metrics.exists())
            self.assertTrue(superalpha.exists())
            self.assertTrue(catalog.exists())
            self.assertGreaterEqual(summary["deleted_dirs"], 2)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_run_health_retention_trims_old_jsonl_lines(self) -> None:
        base_dir = Path("data") / f"_health_retention_test_{uuid.uuid4().hex}"
        universe = "cn_health_retention"
        try:
            cfg = ClosedLoopConfig(
                universe_name=universe,
                universe_base_dir=str(base_dir),
                run_health_retention_enabled=True,
                run_health_retention_max_lines=2,
                run_health_retention_days=0,
            )
            for idx in range(4):
                _append_run_health(
                    config=cfg,
                    result={"status": f"ok_{idx}", "scoreboard_rows": idx},
                    candidate_meta={},
                    selected_meta=pd.DataFrame(),
                    source_chunk_metas=[],
                    retention_summary={},
                    elapsed_seconds=0.1,
                )
            path = base_dir / universe / "feedback" / "run_health.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(len(lines), 2)
            self.assertIn('"status": "ok_2"', lines[0])
            self.assertIn('"status": "ok_3"', lines[1])
            self.assertTrue(path.with_suffix(path.suffix + ".bak").exists())
            self.assertIn("run_health_retention_summary", lines[-1])
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_inspect_closed_loop_health_summarizes_recent_records(self) -> None:
        base_dir = Path("data") / f"_health_inspect_test_{uuid.uuid4().hex}"
        universe = "cn_health_inspect"
        try:
            cfg = ClosedLoopConfig(universe_name=universe, universe_base_dir=str(base_dir))
            _append_run_health(
                config=cfg,
                result={"status": "ok", "scoreboard_rows": 2},
                candidate_meta={},
                selected_meta=pd.DataFrame(),
                source_chunk_metas=[{"mem_warning": True, "mem_mb": 9999.0}],
                retention_summary={
                    "candidate": {"deleted_files": 3},
                    "analysis": {"deleted_dirs": 1},
                },
                elapsed_seconds=0.1,
            )
            script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_closed_loop_health.py"
            spec = importlib.util.spec_from_file_location("inspect_closed_loop_health_under_test", script_path)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)

            summary = module.inspect_closed_loop_health(base_dir=base_dir, universe_name=universe, limit=5)

            self.assertEqual(summary["latest_status"], "ok")
            self.assertEqual(summary["retention_deleted_items"], 4)
            self.assertEqual(summary["memory_warning_count"], 1)
            self.assertEqual(summary["scoreboard_rows_max"], 2)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_validate_universe_registries_reports_backup_recovery(self) -> None:
        base_dir = Path("data") / f"_registry_health_test_{uuid.uuid4().hex}"
        universe = "cn_registry_health"
        try:
            append_universe_expressions(
                pd.DataFrame([{"expression": "rank(close)", "source": "op"}]),
                base_dir=base_dir,
                universe_name=universe,
            )
            append_universe_expressions(
                pd.DataFrame([{"expression": "rank(volume)", "source": "op"}]),
                base_dir=base_dir,
                universe_name=universe,
            )
            expr_path = base_dir / universe / "catalog" / "expressions.csv"
            expr_path.write_text('"unterminated\n1,2,3', encoding="utf-8")

            summary = validate_universe_registries(base_dir=base_dir, universe_name=universe)

            self.assertGreaterEqual(summary["registry_count"], 1)
            self.assertTrue(summary["registry_recovered_from_backup"])
            expression = next(item for item in summary["registries"] if item["name"] == "expression")
            self.assertEqual(expression["status"], "recovered_from_backup")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_run_one_iteration_writes_visualization_png_when_enabled(self) -> None:
        rng = np.random.RandomState(17)
        dates = pd.date_range("2024-01-01", periods=16, freq="D")
        codes = ["A", "B", "C", "D"]
        rows: list[dict[str, object]] = []
        for d in dates:
            for c in codes:
                rows.append(
                    {
                        "date": d,
                        "code": c,
                        "pct_chg": float(rng.normal(0.001, 0.02)),
                        "circ_mv": float(rng.uniform(5e8, 3e9)),
                        "close": float(rng.uniform(10, 200)),
                        "volume": float(rng.uniform(1e5, 5e6)),
                        "industry": "I1" if c in {"A", "B"} else "I2",
                        "sector": "S1" if c in {"A", "C"} else "S2",
                        "universe": 1.0,
                    }
                )
        raw_df = pd.DataFrame(rows)

        base_dir = Path("data") / f"_closed_loop_png_test_{uuid.uuid4().hex}"
        universe = "cn_closed_loop_png_test"
        try:
            cfg = ClosedLoopConfig(
                universe_name=universe,
                universe_base_dir=str(base_dir),
                batch_size=3,
                request_new_alphas=3,
                max_new_alphas_per_chunk=3,
                compute_chunk_size=3,
                max_eval_expressions=60,
                search_mode="operator_only",
                enable_purge_after_analysis=False,
                analysis_layers=4,
                analysis_include_robustness=False,
                include_visualization_png=True,
            )
            out = run_one_loop_iteration(raw_df=raw_df, config=cfg)
            self.assertEqual(out.get("status"), "ok")

            chunk_results = out.get("chunk_results", [])
            self.assertTrue(isinstance(chunk_results, list))
            self.assertGreaterEqual(len(chunk_results), 1)
            meta = chunk_results[0].get("analysis_meta", {})
            table_paths = dict(meta.get("table_paths") or {}) if isinstance(meta, dict) else {}
            self.assertIn("visualization_manifest", table_paths)
            self.assertTrue(Path(table_paths["visualization_manifest"]).exists())
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_trade_date_znz_code_without_pct_chg_is_supported(self) -> None:
        rng = np.random.RandomState(11)
        dates = pd.date_range("2024-02-01", periods=16, freq="D")
        codes = ["A", "B", "C"]
        rows: list[dict[str, object]] = []
        for d in dates:
            for c in codes:
                rows.append(
                    {
                        "trade_date": d,
                        "znz_code": c,
                        "circ_mv": float(rng.uniform(5e8, 2e9)),
                        "close": float(rng.uniform(10, 80)),
                        "volume": float(rng.uniform(1e5, 2e6)),
                        "industry": "I1" if c in {"A", "B"} else "I2",
                        "sector": "S1" if c == "A" else "S2",
                        "universe": 1.0,
                    }
                )
        raw_df = pd.DataFrame(rows)

        base_dir = Path("data") / f"_closed_loop_alias_test_{uuid.uuid4().hex}"
        universe = "cn_closed_loop_alias_test"
        try:
            cfg = ClosedLoopConfig(
                universe_name=universe,
                universe_base_dir=str(base_dir),
                batch_size=3,
                request_new_alphas=3,
                max_new_alphas_per_chunk=3,
                compute_chunk_size=3,
                max_eval_expressions=60,
                search_mode="operator_only",
                enable_purge_after_analysis=False,
                analysis_layers=5,
                analysis_include_robustness=False,
            )
            out = run_one_loop_iteration(raw_df=raw_df, config=cfg)
            self.assertEqual(out.get("status"), "ok")

            base_frame = load_universe_base_frame(base_dir=str(base_dir), universe_name=universe)
            self.assertIn("date", base_frame.columns)
            self.assertIn("code", base_frame.columns)
            self.assertIn("pct_chg", base_frame.columns)
            self.assertGreater(int(base_frame["pct_chg"].notna().sum()), 0)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_exclude_fields_are_not_used_in_generated_expressions(self) -> None:
        rng = np.random.RandomState(23)
        dates = pd.date_range("2024-03-01", periods=20, freq="D")
        codes = ["A", "B", "C", "D"]
        rows: list[dict[str, object]] = []
        for d in dates:
            for c in codes:
                rows.append(
                    {
                        "date": d,
                        "code": c,
                        "pct_chg": float(rng.normal(0.001, 0.02)),
                        "circ_mv": float(rng.uniform(5e8, 3e9)),
                        "close": float(rng.uniform(10, 100)),
                        "volume": float(rng.uniform(1e5, 3e6)),
                        "industry": "I1" if c in {"A", "B"} else "I2",
                        "sector": "S1" if c in {"A", "C"} else "S2",
                        "universe": 1.0,
                    }
                )
        raw_df = pd.DataFrame(rows)

        base_dir = Path("data") / f"_closed_loop_exclude_test_{uuid.uuid4().hex}"
        universe = "cn_closed_loop_exclude_test"
        try:
            cfg = ClosedLoopConfig(
                universe_name=universe,
                universe_base_dir=str(base_dir),
                batch_size=3,
                request_new_alphas=3,
                max_new_alphas_per_chunk=3,
                compute_chunk_size=3,
                max_eval_expressions=80,
                search_mode="operator_only",
                exclude_fields=("close", "volume"),
                enable_purge_after_analysis=False,
                analysis_layers=5,
                analysis_include_robustness=False,
            )
            out = run_one_loop_iteration(raw_df=raw_df, config=cfg)
            self.assertEqual(out.get("status"), "ok")

            reg = load_universe_expression_registry(base_dir=str(base_dir), universe_name=universe)
            self.assertFalse(reg.empty)
            exprs = reg["expression"].astype(str).tolist()
            self.assertTrue(all("close" not in e for e in exprs))
            self.assertTrue(all("volume" not in e for e in exprs))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
