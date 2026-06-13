from __future__ import annotations

import unittest

from alpha_mining.mining.explore import DeepExploreConfig, build_operator_search_space


class TestDeepExplore(unittest.TestCase):
    def test_operator_space_basic(self) -> None:
        cfg = DeepExploreConfig(
            windows=(5, 10),
            max_depth=2,
            max_candidates=80,
            max_inputs_per_operator=8,
            max_binary_pairs=8,
            random_seed=7,
        )
        out = build_operator_search_space(
            available_fields={"close", "volume"},
            available_groups={"industry"},
            config=cfg,
        )
        self.assertTrue(len(out) > 0)
        exprs = [expr for _, expr in out]
        self.assertTrue(any("ts_rank(" in e for e in exprs))
        self.assertTrue(any("group_rank(" in e for e in exprs))
        self.assertTrue(any(" + " in e or " - " in e for e in exprs))

    def test_operator_space_deterministic(self) -> None:
        cfg = DeepExploreConfig(
            windows=(5, 10, 22),
            max_depth=2,
            max_candidates=60,
            max_inputs_per_operator=6,
            max_binary_pairs=6,
            random_seed=123,
        )
        out1 = build_operator_search_space(
            available_fields={"close", "volume"},
            available_groups={"industry"},
            config=cfg,
        )
        out2 = build_operator_search_space(
            available_fields={"close", "volume"},
            available_groups={"industry"},
            config=cfg,
        )
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
