from __future__ import annotations

import shutil
import uuid
import unittest
from pathlib import Path

import pandas as pd

from alpha_mining.hashing import expression_hash
from alpha_mining.history.registry import (
    filter_new_expressions,
    load_seen_hashes_for_config,
    save_run_registry,
)


class TestHistoryRegistry(unittest.TestCase):
    def test_filter_new_expressions(self) -> None:
        exprs = ["rank(close)", "ts_rank(close, 5)", "rank(close)"]
        seen = {expression_hash("rank(close)")}
        new_exprs, skipped = filter_new_expressions(exprs, seen_hashes=seen)
        self.assertEqual(new_exprs, ["ts_rank(close, 5)"])
        self.assertEqual(skipped, 2)

    def test_save_and_load_registry(self) -> None:
        base = Path("data") / f"_registry_test_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            cfg = {
                "search_mode": "deep_hybrid",
                "max_eval": 100,
                "pool": {"TPL-001": {"field": ["close"], "d": [5, 10]}},
            }
            expr_df = pd.DataFrame(
                {
                    "alpha_name": ["alpha_0001", "alpha_0002"],
                    "expression": ["rank(close)", "ts_rank(close, 5)"],
                    "source": ["template:TPL-002", "template:TPL-001"],
                }
            )

            rec = save_run_registry(base_dir=base, config_snapshot=cfg, expression_df=expr_df)
            self.assertEqual(rec["expression_count"], 2)
            cfg_hash = rec["config_hash"]
            seen = load_seen_hashes_for_config(base, cfg_hash)
            self.assertEqual(len(seen), 2)

            # Second run with one repeated expression should not increase unique count by 2.
            expr_df2 = pd.DataFrame(
                {
                    "alpha_name": ["alpha_0001", "alpha_0002"],
                    "expression": ["rank(close)", "rank(volume)"],
                    "source": ["x", "x"],
                }
            )
            rec2 = save_run_registry(base_dir=base, config_snapshot=cfg, expression_df=expr_df2)
            seen2 = load_seen_hashes_for_config(base, cfg_hash)
            self.assertTrue(rec2["config_total_unique_expressions"] >= 3)
            self.assertEqual(len(seen2), rec2["config_total_unique_expressions"])
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
