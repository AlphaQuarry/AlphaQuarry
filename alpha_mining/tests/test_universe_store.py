from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from alpha_mining.workflow import (
    append_universe_expressions,
    append_alpha_lifecycle_records,
    delete_universe_alpha_values,
    init_universe_workspace,
    load_alpha_lifecycle_registry,
    load_universe_alpha_batch,
    load_universe_analysis_registry,
    load_universe_base_frame,
    load_universe_expression_registry,
    load_universe_input_manifest,
    load_factor_metrics_registry,
    save_universe_alpha_values,
    save_universe_analysis_run,
    save_universe_base_frame,
    save_universe_input_manifest,
    select_alpha_names_for_analysis,
    update_alpha_lifecycle_status,
)
from alpha_mining.mining.candidate_planner import (
    prune_candidate_artifacts,
    save_candidate_artifacts,
)
from alpha_mining.workflow.universe_store import build_dashboard_factor_metrics


class TestUniverseStore(unittest.TestCase):
    def test_universe_expression_and_alpha_flow(self) -> None:
        base_dir = Path("data") / f"_universe_store_test_{uuid.uuid4().hex}"
        universe_name = "cn_test"
        try:
            init_universe_workspace(base_dir=base_dir, universe_name=universe_name)

            base_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                    "code": ["A", "B", "A", "B"],
                    "pct_chg": [0.01, -0.02, 0.03, -0.01],
                    "circ_mv": [1e9, 2e9, 1.1e9, 2.1e9],
                }
            )
            save_universe_base_frame(base_df=base_df, base_dir=base_dir, universe_name=universe_name)
            loaded_base = load_universe_base_frame(base_dir=base_dir, universe_name=universe_name)
            self.assertEqual(len(loaded_base), len(base_df))

            expr_df = pd.DataFrame(
                {
                    "expression": ["rank(close)", "ts_rank(close, 5)", "rank(close)"],
                    "source": ["op", "op", "op"],
                }
            )
            added = append_universe_expressions(expression_df=expr_df, base_dir=base_dir, universe_name=universe_name)
            self.assertEqual(len(added), 2)
            self.assertEqual(added["alpha_name"].tolist(), ["alpha00001", "alpha00002"])

            reg = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
            self.assertEqual(len(reg), 2)
            self.assertIn("simulation_config_json", reg.columns)
            self.assertIn("input_manifest_id", reg.columns)

            saved_manifest = save_universe_input_manifest(
                manifest={
                    "source_path": "data/raw.pkl",
                    "date_col": "date",
                    "code_col": "code",
                },
                base_dir=base_dir,
                universe_name=universe_name,
            )
            loaded_manifest = load_universe_input_manifest(
                base_dir=base_dir,
                universe_name=universe_name,
                manifest_id=saved_manifest["manifest_id"],
            )
            self.assertTrue(isinstance(loaded_manifest, dict))
            self.assertEqual(loaded_manifest.get("manifest_id"), saved_manifest["manifest_id"])

            alpha_1 = base_df[["date", "code"]].copy()
            alpha_1["alpha00001"] = [0.1, -0.1, 0.2, -0.2]
            alpha_2 = base_df[["date", "code"]].copy()
            alpha_2["alpha00002"] = [0.3, -0.3, 0.4, -0.4]
            save_universe_alpha_values(
                alpha_df=alpha_1,
                alpha_name="alpha00001",
                base_dir=base_dir,
                universe_name=universe_name,
            )
            save_universe_alpha_values(
                alpha_df=alpha_2,
                alpha_name="alpha00002",
                base_dir=base_dir,
                universe_name=universe_name,
            )

            alpha_batch = load_universe_alpha_batch(
                alpha_names=["alpha00001", "alpha00002"],
                base_dir=base_dir,
                universe_name=universe_name,
            )
            self.assertIn("alpha00001", alpha_batch.columns)
            self.assertIn("alpha00002", alpha_batch.columns)
            self.assertEqual(len(alpha_batch), len(base_df))

            selected = select_alpha_names_for_analysis(
                base_dir=base_dir,
                universe_name=universe_name,
                mode="next_pending",
                batch_size=5,
                period=1,
                layers=10,
                is_timeseries=True,
                force_reanalyze=False,
            )
            self.assertEqual(selected, ["alpha00001", "alpha00002"])

            factor_metrics = pd.DataFrame(
                {
                    "factor": ["alpha00001", "alpha00002"],
                    "ic_mean": [0.03, 0.02],
                    "ir": [0.6, 0.3],
                    "long_short_total_return": [0.10, 0.05],
                    "long_short_annualized_return": [0.12, 0.06],
                    "long_short_volatility": [0.20, 0.30],
                    "long_short_sharpe_ratio": [1.2, 0.6],
                    "long_short_max_drawdown": [0.10, 0.20],
                    "long_short_fitness_ratio": [0.5, 0.2],
                    "best_layer_total_return": [0.09, 0.04],
                    "best_layer_annualized_return": [0.11, 0.05],
                    "best_layer_volatility": [0.18, 0.25],
                    "best_layer_sharpe": [1.1, 0.5],
                    "best_layer_max_drawdown": [0.08, 0.18],
                    "best_layer_fitness_ratio": [0.4, 0.1],
                    "best_minus_universe_annualized_return": [0.04, 0.01],
                    "turnover_long_only_mean": [0.3, 0.4],
                    "margin_long_only": [0.002, 0.001],
                    "score_predictive_power": [70.0, 50.0],
                    "score_long_only_performance": [80.0, 40.0],
                    "score_stability": [60.0, 30.0],
                    "score_tradeability": [90.0, 20.0],
                    "score_total": [75.0, 35.0],
                    "effectiveness_tier": ["A", "C"],
                }
            )
            dashboard_metrics = build_dashboard_factor_metrics(
                factor_metrics_df=factor_metrics,
                expression_registry_df=reg,
                period=1,
                layers=10,
            )
            legacy_columns = [
                "factor",
                "period",
                "layers",
                "expression",
                "ic_mean",
                "ir",
                "long_short_total_return",
                "long_short_annualized_return",
                "long_short_volatility",
                "long_short_sharpe_ratio",
                "long_short_max_drawdown",
                "long_short_fitness_ratio",
                "best_layer_total_return",
                "best_layer_annualized_return",
                "best_layer_volatility",
                "best_layer_sharpe",
                "best_layer_max_drawdown",
                "best_layer_fitness_ratio",
                "best_minus_universe_annualized_return",
                "turnover_long_only_mean",
                "margin_long_only",
                "score_predictive_power",
                "score_long_only_performance",
                "score_stability",
                "score_tradeability",
                "score_total",
                "effectiveness_tier",
            ]
            self.assertEqual(
                dashboard_metrics.columns.tolist()[: len(legacy_columns)],
                legacy_columns,
            )
            self.assertIn("feedback_score", dashboard_metrics.columns)
            self.assertIn("train_score_total", dashboard_metrics.columns)
            self.assertIn("val_score_total", dashboard_metrics.columns)
            self.assertIn("test_score_total", dashboard_metrics.columns)
            self.assertEqual(dashboard_metrics.loc[0, "expression"], "rank(close)")

            analysis_meta = save_universe_analysis_run(
                base_dir=base_dir,
                universe_name=universe_name,
                alpha_names=selected,
                period=1,
                layers=10,
                is_timeseries=True,
                factor_metrics_df=factor_metrics,
                tables={
                    "summary_df": factor_metrics,
                    "portfolio_pnl_df": pd.DataFrame(
                        {
                            "factor": ["alpha00001"],
                            "trade_date": pd.to_datetime(["2024-01-01"]),
                            "portfolio": ["long_10"],
                            "return": [0.01],
                            "cum_return": [0.01],
                        }
                    ),
                    "dashboard_factor_metrics": dashboard_metrics,
                },
            )
            self.assertTrue(analysis_meta["table_paths"]["portfolio_pnl_df"].endswith(".parquet"))
            self.assertTrue(analysis_meta["table_paths"]["dashboard_factor_metrics"].endswith(".csv"))
            self.assertTrue(Path(analysis_meta["table_paths"]["portfolio_pnl_df"]).exists())
            self.assertTrue(Path(analysis_meta["table_paths"]["dashboard_factor_metrics"]).exists())

            analyzed = load_universe_analysis_registry(base_dir=base_dir, universe_name=universe_name)
            self.assertEqual(len(analyzed), 2)
            metrics_registry = load_factor_metrics_registry(base_dir=base_dir, universe_name=universe_name)
            self.assertGreaterEqual(len(metrics_registry), 1)

            selected_after = select_alpha_names_for_analysis(
                base_dir=base_dir,
                universe_name=universe_name,
                mode="next_pending",
                batch_size=5,
                period=1,
                layers=10,
                is_timeseries=True,
                force_reanalyze=False,
            )
            self.assertEqual(selected_after, [])

            append_alpha_lifecycle_records(
                pd.DataFrame(
                    {
                        "alpha_name": ["alpha00001"],
                        "expression": ["rank(close)"],
                        "expression_hash": ["h1"],
                        "source": ["op"],
                        "status": ["REGISTERED"],
                    }
                ),
                base_dir=base_dir,
                universe_name=universe_name,
            )
            update_alpha_lifecycle_status(
                alpha_names=["alpha00001"],
                status="MATERIALIZED",
                alpha_value_path="path/to/value.parquet",
                base_dir=base_dir,
                universe_name=universe_name,
            )
            lifecycle = load_alpha_lifecycle_registry(base_dir=base_dir, universe_name=universe_name)
            self.assertEqual(len(lifecycle), 1)
            self.assertEqual(lifecycle.iloc[0]["status"], "MATERIALIZED")
            self.assertIn("failure_kind", lifecycle.columns)
            self.assertIn("last_error_stage", lifecycle.columns)

            purge_result = delete_universe_alpha_values(
                alpha_names=["alpha00001"],
                base_dir=base_dir,
                universe_name=universe_name,
            )
            self.assertEqual(purge_result["requested"], 1)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_registry_load_falls_back_to_backup_when_current_csv_is_corrupt(
        self,
    ) -> None:
        base_dir = Path("data") / f"_universe_store_backup_test_{uuid.uuid4().hex}"
        universe_name = "cn_backup"
        try:
            append_universe_expressions(
                pd.DataFrame([{"expression": "rank(close)", "source": "op"}]),
                base_dir=base_dir,
                universe_name=universe_name,
            )
            append_universe_expressions(
                pd.DataFrame([{"expression": "rank(volume)", "source": "op"}]),
                base_dir=base_dir,
                universe_name=universe_name,
            )
            path = base_dir / universe_name / "catalog" / "expressions.csv"
            self.assertTrue(path.with_suffix(path.suffix + ".bak").exists())
            path.write_text('"unterminated\n1,2,3', encoding="utf-8")

            loaded = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)

            self.assertFalse(loaded.empty)
            self.assertIn("rank(close)", set(loaded["expression"].astype(str)))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_candidate_artifact_retention_prunes_old_batches_only(self) -> None:
        base_dir = Path("data") / f"_candidate_retention_test_{uuid.uuid4().hex}"
        root = base_dir / "u1"
        try:
            for batch_id in ["batch_1000", "batch_2000", "batch_3000"]:
                save_candidate_artifacts(
                    candidate_df=pd.DataFrame([{"candidate_id": batch_id, "expression": "rank(close)"}]),
                    rejected_df=pd.DataFrame([{"candidate_id": batch_id, "expression": "bad"}]),
                    root=root,
                    batch_id=batch_id,
                    generation_diagnostics={"batch": batch_id},
                )
            keep_file = root / "catalog" / "expressions.csv"
            keep_file.parent.mkdir(parents=True, exist_ok=True)
            keep_file.write_text("alpha_name,expression\nalpha00001,rank(close)\n", encoding="utf-8")

            summary = prune_candidate_artifacts(root=root, max_batches=1, retention_days=0)

            files = {p.name for p in (root / "catalog" / "candidates").iterdir()}
            self.assertTrue(any(name.startswith("batch_3000") for name in files))
            self.assertFalse(any(name.startswith("batch_1000") for name in files))
            self.assertTrue(keep_file.exists())
            self.assertGreater(summary["deleted_files"], 0)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
