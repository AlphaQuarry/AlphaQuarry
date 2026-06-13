from __future__ import annotations

import unittest
from decimal import Decimal

import pandas as pd

from alpha_mining.operators.group_ops import (
    _bucket,
    _group_cartesian_product,
    _group_neutralize,
    _group_rank,
)


class TestGroupOps(unittest.TestCase):
    def test_group_rank_shape(self) -> None:
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cols = ["A", "B", "C"]
        x = pd.DataFrame([[1, 2, 3], [2, 1, 3]], index=idx, columns=cols, dtype=float)
        g = pd.DataFrame([["G1", "G1", "G2"], ["G1", "G1", "G2"]], index=idx, columns=cols)
        out = _group_rank(x, g)
        self.assertEqual(out.shape, x.shape)

    def test_group_neutralize_decimal_input(self) -> None:
        idx = pd.to_datetime(["2024-01-01"])
        cols = ["A", "B", "C"]
        x = pd.DataFrame(
            [[Decimal("1.0"), Decimal("2.0"), Decimal("3.0")]],
            index=idx,
            columns=cols,
            dtype=object,
        )
        g = pd.DataFrame([["G1", "G1", "G2"]], index=idx, columns=cols)
        out = _group_neutralize(x, g)
        self.assertTrue(pd.api.types.is_float_dtype(out["A"]))

    def test_bucket_invalid_range_returns_nan_panel(self) -> None:
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cols = ["A", "B"]
        x = pd.DataFrame([[0.1, 0.9], [0.2, 0.8]], index=idx, columns=cols, dtype=float)

        for range_text in ["bad", "0,1,0", "1,0,0.2", "0,1"]:
            out = _bucket(x, range_text)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(out.isna().all().all())

    def test_group_cartesian_product_preserves_nan(self) -> None:
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cols = ["A", "B", "C"]
        left = pd.DataFrame([["bank", None, "tech"], ["bank", "steel", None]], index=idx, columns=cols)
        right = pd.DataFrame([[0, 1, None], [None, 2, 3]], index=idx, columns=cols)

        out = _group_cartesian_product(left, right)

        self.assertTrue(str(out.loc[idx[0], "A"]).startswith("bank__"))
        self.assertTrue(pd.isna(out.loc[idx[0], "B"]))
        self.assertTrue(pd.isna(out.loc[idx[0], "C"]))
        self.assertTrue(pd.isna(out.loc[idx[1], "A"]))
        self.assertTrue(str(out.loc[idx[1], "B"]).startswith("steel__"))
        self.assertTrue(pd.isna(out.loc[idx[1], "C"]))


if __name__ == "__main__":
    unittest.main()
