from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd

import alpha_mining.workflow.artifacts as artifacts_module
from alpha_mining.workflow import (
    compile_feedback_scoreboard,
    init_run_workspace,
    load_analysis_manifest,
    load_base_frame,
    load_mining_manifest,
    save_analysis_batch,
    save_base_frame,
    save_mining_batch,
)
from alpha_mining.workflow.artifacts import save_dataframe_artifact


class TestWorkflowArtifacts(unittest.TestCase):
    def test_run_batch_save_and_feedback(self) -> None:
        base_dir = Path("data") / f"_workflow_test_{uuid.uuid4().hex}"
        try:
            run = init_run_workspace(
                base_dir=base_dir,
                config_snapshot={"search_mode": "template_only", "batch_size": 10},
                extra_meta={"test_case": "workflow"},
            )
            run_dir = Path(run["run_dir"])

            base_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                    "code": ["A", "B", "A", "B"],
                    "pct_chg": [0.01, -0.02, 0.03, 0.01],
                    "circ_mv": [1e9, 2e9, 1.1e9, 2.1e9],
                }
            )
            base_saved = save_base_frame(run_dir, base_df)
            self.assertTrue(Path(base_saved["path"]).exists())
            loaded_base = load_base_frame(run_dir)
            self.assertEqual(len(loaded_base), len(base_df))

            alpha_df = pd.DataFrame(
                {
                    "date": base_df["date"],
                    "code": base_df["code"],
                    "alpha_0001": [0.1, -0.1, 0.2, -0.2],
                    "alpha_0002": [0.3, -0.3, 0.4, -0.4],
                }
            )
            expr_df = pd.DataFrame(
                {
                    "alpha_name": ["alpha_0001", "alpha_0002"],
                    "expression": ["rank(close)", "ts_rank(close, 5)"],
                    "source": ["template:TPL-002", "template:TPL-001"],
                }
            )
            save_mining_batch(
                run_dir=run_dir,
                batch_index=1,
                alpha_start=1,
                alpha_end=2,
                alpha_df=alpha_df,
                expression_df=expr_df,
            )
            mining_manifest = load_mining_manifest(run_dir)
            self.assertEqual(len(mining_manifest), 1)
            self.assertIn("batch_001__alpha_0001-0002", mining_manifest["batch_id"].tolist())

            factor_metrics_df = pd.DataFrame(
                {
                    "factor": ["alpha_0001", "alpha_0002"],
                    "ic_mean": [0.04, 0.01],
                    "ir": [0.7, 0.2],
                    "long_short_sharpe_ratio": [1.2, 0.3],
                    "long_short_total_return": [0.3, 0.05],
                    "avg_turnover": [0.2, 0.4],
                }
            )
            save_analysis_batch(
                run_dir=run_dir,
                batch_id="batch_001__alpha_0001-0002",
                period=1,
                factor_metrics_df=factor_metrics_df,
                tables={"summary_df": factor_metrics_df[["factor", "ic_mean", "ir"]]},
            )
            analysis_manifest = load_analysis_manifest(run_dir)
            self.assertEqual(len(analysis_manifest), 1)
            self.assertEqual(int(analysis_manifest.iloc[0]["period"]), 1)

            scoreboard = compile_feedback_scoreboard(run_dir=run_dir)
            self.assertEqual(len(scoreboard), 2)
            self.assertIn("expression", scoreboard.columns)
            self.assertIn("composite_score", scoreboard.columns)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


def test_save_dataframe_artifact_writes_parquet_via_temp_path_then_replaces(
    tmp_path: Path,
) -> None:
    observed_paths: list[Path] = []

    def fake_write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
        p = Path(path)
        observed_paths.append(p)
        p.write_bytes(b"parquet-bytes")

    with mock.patch.object(artifacts_module, "write_parquet_compat", side_effect=fake_write_parquet):
        saved = save_dataframe_artifact(pd.DataFrame({"x": [1]}), tmp_path / "artifact", preferred="parquet")

    final_path = tmp_path / "artifact.parquet"
    assert saved == {"path": final_path.as_posix(), "format": "parquet"}
    assert final_path.read_bytes() == b"parquet-bytes"
    assert observed_paths
    assert observed_paths[0] != final_path
    assert observed_paths[0].parent == final_path.parent


def test_save_dataframe_artifact_keeps_existing_parquet_when_temp_write_fails(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "artifact.parquet"
    final_path.write_bytes(b"old-good")

    def fake_write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("boom")

    with mock.patch.object(artifacts_module, "write_parquet_compat", side_effect=fake_write_parquet):
        saved = save_dataframe_artifact(pd.DataFrame({"x": [1]}), tmp_path / "artifact", preferred="parquet")

    assert final_path.read_bytes() == b"old-good"
    assert saved["format"] == "pickle"
    assert Path(saved["path"]).exists()
