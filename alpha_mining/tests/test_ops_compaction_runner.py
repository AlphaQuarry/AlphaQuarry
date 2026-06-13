from __future__ import annotations

import importlib.util
import gc
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_module"), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestOpsCompactionRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compact_mod = _load_script("compact_parquet_lake.py")
        cls.runner_mod = _load_script("repo_ops_runner.py")

    def test_compact_table_root_merges_month_partition(self) -> None:
        base_dir = Path("data") / f"_ops_compact_{uuid.uuid4().hex}"
        partition = base_dir / "facts" / "market_daily" / "year=2026" / "month=04"
        try:
            partition.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2026-04-01"], "code": ["000001.SZ"]}).to_parquet(
                partition / "part-001.parquet", index=False
            )
            pd.DataFrame({"date": ["2026-04-02"], "code": ["000002.SZ"]}).to_parquet(
                partition / "part-002.parquet", index=False
            )
            gc.collect()

            summary = self.compact_mod.compact_table_root(table_root=base_dir / "facts" / "market_daily", dry_run=False)

            self.assertEqual(int(summary["partitions_compacted"]), 1)
            self.assertGreaterEqual(int(summary["files_removed"]) + int(summary["files_remove_failed"]), 1)
            files = sorted(partition.glob("*.parquet"))
            self.assertIn("part-000.parquet", [p.name for p in files])
            merged = pd.read_parquet(partition / "part-000.parquet")
            self.assertEqual(len(merged), 2)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_build_jobs_contains_selected_commands(self) -> None:
        jobs = self.runner_mod.build_jobs(
            config="configs/datasource.local.yaml",
            selected_jobs=["rebuild_duckdb_catalog", "monthly_compaction"],
            catalog_args=["--source-view", "v_project_panel_cn_a"],
            compaction_args=["--dry-run"],
        )

        self.assertEqual([job.name for job in jobs], ["rebuild_duckdb_catalog", "monthly_compaction"])
        self.assertIn("build_duckdb_catalog.py", " ".join(jobs[0].command))
        self.assertIn("--source-view", jobs[0].command)
        self.assertIn("compact_parquet_lake.py", " ".join(jobs[1].command))
        self.assertIn("--dry-run", jobs[1].command)


if __name__ == "__main__":
    unittest.main()
