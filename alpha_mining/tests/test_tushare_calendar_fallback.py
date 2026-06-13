from __future__ import annotations

import unittest

import pandas as pd

from alpha_mining.datasource.tushare_client import TushareClient


class TestTushareCalendarFallback(unittest.TestCase):
    def test_bundle_uses_trade_cal_when_available(self) -> None:
        client = TushareClient.__new__(TushareClient)

        def _fetch_trade_cal(*args, **kwargs):  # type: ignore[no-untyped-def]
            return pd.DataFrame(
                {
                    "exchange": ["SSE", "SSE"],
                    "cal_date": ["20260401", "20260402"],
                    "is_open": [1, 1],
                    "pretrade_date": ["20260331", "20260401"],
                }
            )

        def _probe(*args, **kwargs):  # type: ignore[no-untyped-def]
            return ["20260401"]

        client.fetch_trade_cal = _fetch_trade_cal  # type: ignore[method-assign]
        client._probe_open_trade_dates_by_daily = _probe  # type: ignore[method-assign]

        cal, open_dates, source = client.fetch_trade_calendar_bundle("2026-04-01", "2026-04-02", "SSE")
        self.assertEqual(source, "trade_cal")
        self.assertEqual(open_dates, ["20260401", "20260402"])
        self.assertEqual(len(cal), 2)

    def test_bundle_falls_back_to_daily_probe(self) -> None:
        client = TushareClient.__new__(TushareClient)

        def _fetch_trade_cal(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("trade_cal disabled")

        def _probe(*args, **kwargs):  # type: ignore[no-untyped-def]
            return ["20260401", "20260402"]

        client.fetch_trade_cal = _fetch_trade_cal  # type: ignore[method-assign]
        client._probe_open_trade_dates_by_daily = _probe  # type: ignore[method-assign]

        cal, open_dates, source = client.fetch_trade_calendar_bundle("2026-04-01", "2026-04-02", "SSE")
        self.assertEqual(source, "daily_probe")
        self.assertEqual(open_dates, ["20260401", "20260402"])
        self.assertIn("cal_date", cal.columns)
        self.assertEqual(len(cal), 2)


if __name__ == "__main__":
    unittest.main()
