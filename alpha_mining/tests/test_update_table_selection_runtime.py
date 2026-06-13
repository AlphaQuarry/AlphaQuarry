from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def _load_update_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "update_tushare_lake.py"
    spec = importlib.util.spec_from_file_location("update_tushare_lake_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load update_tushare_lake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bootstrap_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_tushare_lake.py"
    spec = importlib.util.spec_from_file_location("bootstrap_tushare_lake_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load bootstrap_tushare_lake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {
            "daily": 0,
            "daily_basic": 0,
            "adj_factor": 0,
            "stk_limit": 0,
            "suspend_d": 0,
            "income_vip": 0,
            "balancesheet_vip": 0,
            "cashflow_vip": 0,
            "fina_indicator_vip": 0,
            "moneyflow_ths": 0,
            "ths_index": 0,
            "ths_member": 0,
        }
        self.fail_on: set[str] = set()

    def fetch_daily_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "daily" in self.fail_on:
            raise AssertionError("daily should not be called in this scenario")
        self.calls["daily"] += 1
        return pd.DataFrame({"trade_date": [trade_date], "ts_code": ["000001.SZ"], "close": [10.0]})

    def fetch_daily_basic_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "daily_basic" in self.fail_on:
            raise AssertionError("daily_basic should not be called in this scenario")
        self.calls["daily_basic"] += 1
        return pd.DataFrame({"trade_date": [trade_date], "ts_code": ["000001.SZ"], "circ_mv": [1.0]})

    def fetch_adj_factor_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "adj_factor" in self.fail_on:
            raise AssertionError("adj_factor should not be called in this scenario")
        self.calls["adj_factor"] += 1
        return pd.DataFrame({"trade_date": [trade_date], "ts_code": ["000001.SZ"], "adj_factor": [1.0]})

    def fetch_stk_limit_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "stk_limit" in self.fail_on:
            raise AssertionError("stk_limit should not be called in this scenario")
        self.calls["stk_limit"] += 1
        return pd.DataFrame({"trade_date": [trade_date], "ts_code": ["000001.SZ"], "up_limit": [11.0]})

    def fetch_suspend_d_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "suspend_d" in self.fail_on:
            raise AssertionError("suspend_d should not be called in this scenario")
        self.calls["suspend_d"] += 1
        return pd.DataFrame({"trade_date": [trade_date], "ts_code": ["000001.SZ"], "suspend_type": [""]})

    def fetch_moneyflow_ths_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        if "moneyflow_ths" in self.fail_on:
            raise AssertionError("moneyflow_ths should not be called in this scenario")
        self.calls["moneyflow_ths"] += 1
        return pd.DataFrame(
            {
                "trade_date": [trade_date],
                "ts_code": ["000001.SZ"],
                "net_amount": [1000.0],
                "net_d5_amount": [2000.0],
            }
        )

    def fetch_ths_index(self) -> pd.DataFrame:
        if "ths_index" in self.fail_on:
            raise AssertionError("ths_index should not be called in this scenario")
        self.calls["ths_index"] += 1
        return pd.DataFrame({"ts_code": ["885001.TI"], "name": ["theme"], "count": [10]})

    def fetch_ths_member(self, ts_codes=None, con_code=None) -> pd.DataFrame:
        _ = con_code
        if "ths_member" in self.fail_on:
            raise AssertionError("ths_member should not be called in this scenario")
        self.calls["ths_member"] += 1
        codes = [str(x) for x in (ts_codes or ["885001.TI"])]
        return pd.DataFrame(
            {
                "ts_code": codes,
                "con_code": ["000001.SZ"] * len(codes),
                "con_name": ["PINGAN"] * len(codes),
                "in_date": ["20260410"] * len(codes),
            }
        )

    def fetch_open_trade_dates(self, start_date: str, end_date: str, exchange: str) -> list[str]:
        return ["20260408", "20260409", "20260410"]

    def fetch_income_vip(self, start_date: str, end_date: str) -> pd.DataFrame:
        _ = (start_date, end_date)
        if "income_vip" in self.fail_on:
            raise AssertionError("income_vip should not be called in this scenario")
        self.calls["income_vip"] += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260410"],
                "end_date": ["20260331"],
                "total_revenue": [100.0],
                "revenue": [90.0],
                "n_income_attr_p": [10.0],
            }
        )

    def fetch_balancesheet_vip(self, start_date: str, end_date: str) -> pd.DataFrame:
        _ = (start_date, end_date)
        if "balancesheet_vip" in self.fail_on:
            raise AssertionError("balancesheet_vip should not be called in this scenario")
        self.calls["balancesheet_vip"] += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260410"],
                "end_date": ["20260331"],
                "total_assets": [1000.0],
                "total_liab": [500.0],
            }
        )

    def fetch_cashflow_vip(self, start_date: str, end_date: str) -> pd.DataFrame:
        _ = (start_date, end_date)
        if "cashflow_vip" in self.fail_on:
            raise AssertionError("cashflow_vip should not be called in this scenario")
        self.calls["cashflow_vip"] += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260410"],
                "end_date": ["20260331"],
                "n_cashflow_act": [50.0],
            }
        )

    def fetch_fina_indicator_vip(self, start_date: str, end_date: str) -> pd.DataFrame:
        _ = (start_date, end_date)
        if "fina_indicator_vip" in self.fail_on:
            raise AssertionError("fina_indicator_vip should not be called in this scenario")
        self.calls["fina_indicator_vip"] += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260410"],
                "end_date": ["20260331"],
                "roe": [0.1],
                "roa": [0.05],
            }
        )


class _FakeLake:
    def __init__(self, payload: dict, inferred_max: dict[str, str] | None = None):
        self._payload = payload
        self._inferred_max = dict(inferred_max or {})

    def load_ingestion_state(self) -> dict:
        return self._payload

    def infer_vendor_table_max_trade_date(self, table: str, date_col: str = "trade_date") -> str:
        _ = date_col
        return str(self._inferred_max.get(str(table), ""))


class TestUpdateTableSelectionRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_update_module()

    def test_fetch_fact_frame_uses_selected_table(self) -> None:
        client = _FakeClient()
        out = self.mod._fetch_fact_frame(client, "stk_limit", "20260410")
        self.assertIn("up_limit", out.columns)
        self.assertEqual(len(out), 1)
        self.assertEqual(client.calls["stk_limit"], 1)

    def test_fetch_selected_fact_frames_p1_only_does_not_touch_p0(self) -> None:
        client = _FakeClient()
        client.fail_on = {"daily", "daily_basic", "adj_factor"}
        out = self.mod._fetch_selected_fact_frames(
            client=client,
            selected_fact_tables=["stk_limit", "suspend_d"],
            trade_date="20260410",
        )
        self.assertEqual(set(out.keys()), {"stk_limit", "suspend_d"})
        self.assertEqual(client.calls["stk_limit"], 1)
        self.assertEqual(client.calls["suspend_d"], 1)
        self.assertEqual(client.calls["daily"], 0)
        self.assertEqual(client.calls["daily_basic"], 0)
        self.assertEqual(client.calls["adj_factor"], 0)

    def test_fetch_selected_fact_frames_p3_moneyflow_only(self) -> None:
        client = _FakeClient()
        client.fail_on = {
            "daily",
            "daily_basic",
            "adj_factor",
            "stk_limit",
            "suspend_d",
        }
        out = self.mod._fetch_selected_fact_frames(
            client=client,
            selected_fact_tables=["moneyflow_ths"],
            trade_date="20260410",
        )
        self.assertEqual(set(out.keys()), {"moneyflow_ths"})
        self.assertEqual(client.calls["moneyflow_ths"], 1)
        self.assertEqual(client.calls["daily"], 0)

    def test_resolve_default_start_uses_anchor_tables(self) -> None:
        state = {
            "tables": {
                "daily": {"last_trade_date": "2026-04-20"},
                "stk_limit": {"last_trade_date": "2026-04-10"},
                "suspend_d": {"last_trade_date": "2026-04-15"},
            }
        }
        lake = _FakeLake(state)
        client = _FakeClient()
        out = self.mod._resolve_default_start_date(
            lake=lake,
            client=client,
            end_date="2026-04-22",
            exchange="SSE",
            lookback_trade_days=0,
            anchor_tables=["stk_limit", "suspend_d"],
        )
        self.assertEqual(out, "2026-04-11")

    def test_resolve_default_start_falls_back_to_daily_when_anchor_missing(
        self,
    ) -> None:
        state = {
            "tables": {
                "daily": {"last_trade_date": "2026-04-20"},
            }
        }
        lake = _FakeLake(state)
        client = _FakeClient()
        out = self.mod._resolve_default_start_date(
            lake=lake,
            client=client,
            end_date="2026-04-22",
            exchange="SSE",
            lookback_trade_days=0,
            anchor_tables=["stk_limit", "suspend_d"],
        )
        self.assertEqual(out, "2026-04-21")

    def test_resolve_default_start_uses_lake_max_when_state_is_stale(self) -> None:
        state = {
            "tables": {
                "daily": {"last_trade_date": "2021-04-20"},
            }
        }
        lake = _FakeLake(state, inferred_max={"daily": "2026-04-20"})
        client = _FakeClient()
        out = self.mod._resolve_default_start_date(
            lake=lake,
            client=client,
            end_date="2026-04-22",
            exchange="SSE",
            lookback_trade_days=0,
            anchor_tables=["daily"],
        )
        self.assertEqual(out, "2026-04-21")

    def test_build_execution_plan_estimates_calls(self) -> None:
        plan = self.mod._build_execution_plan(
            selected_trade_fact_tables=["stk_limit", "suspend_d"],
            selected_range_fact_tables=["income_vip"],
            selected_dim_tables=["namechange"],
            refresh_dims=True,
            start_date="2026-04-21",
            end_date="2026-04-21",
            open_trade_dates=["20260420", "20260421"],
            pending_trade_dates=["20260421"],
            flush_trade_days=60,
            range_window_days=60,
            prune_out_of_range=False,
            skip_duckdb=False,
            trade_calendar_source="trade_cal",
            need_trade_bundle=True,
        )
        self.assertEqual(plan["trade_fact_calls_by_table"], {"stk_limit": 1, "suspend_d": 1})
        self.assertEqual(plan["range_fact_calls_by_table"], {"income_vip": 1})
        self.assertEqual(int(plan["range_window_count"]), 1)
        self.assertEqual(plan["dim_calls_by_table"], {"namechange": 1})
        self.assertEqual(int(plan["estimated_total_api_calls_base"]), 5)
        self.assertEqual(int(plan["estimated_total_api_calls_max"]), 7)
        self.assertEqual(bool(plan["rebuild_duckdb_catalog"]), True)

    def test_build_execution_plan_counts_p3_dims_and_moneyflow(self) -> None:
        plan = self.mod._build_execution_plan(
            selected_trade_fact_tables=["moneyflow_ths"],
            selected_range_fact_tables=[],
            selected_dim_tables=["ths_index", "ths_member"],
            refresh_dims=True,
            start_date="2026-04-21",
            end_date="2026-04-21",
            open_trade_dates=["20260421"],
            pending_trade_dates=["20260421"],
            flush_trade_days=60,
            range_window_days=60,
            prune_out_of_range=False,
            skip_duckdb=False,
            trade_calendar_source="trade_cal",
            need_trade_bundle=True,
        )
        self.assertEqual(plan["trade_fact_calls_by_table"], {"moneyflow_ths": 1})
        self.assertEqual(plan["dim_calls_by_table"], {"ths_index": 1, "ths_member": 1})
        self.assertEqual(int(plan["estimated_total_api_calls_base"]), 4)

    def test_refresh_range_facts_p2_only_does_not_touch_trade_day_facts(self) -> None:
        client = _FakeClient()
        client.fail_on = {
            "daily",
            "daily_basic",
            "adj_factor",
            "stk_limit",
            "suspend_d",
        }

        class _StubLake:
            def __init__(self) -> None:
                self.vendor_calls = 0
                self.curated_calls = 0
                self.state_calls = 0

            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                self.vendor_calls += 1
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                self.curated_calls += 1
                return {}

            def update_ingestion_state(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                self.state_calls += 1
                return {}

        lake = _StubLake()
        summary = self.mod._refresh_range_fact_tables(
            lake=lake,
            client=client,
            selected_range_fact_tables=["income_vip", "fina_indicator_vip"],
            start_date="2026-01-01",
            end_date="2026-04-21",
        )
        self.assertEqual(client.calls["income_vip"], 1)
        self.assertEqual(client.calls["fina_indicator_vip"], 1)
        self.assertEqual(client.calls["daily"], 0)
        self.assertEqual(client.calls["daily_basic"], 0)
        self.assertEqual(client.calls["adj_factor"], 0)
        self.assertEqual(lake.vendor_calls, 2)
        self.assertEqual(lake.curated_calls, 2)
        self.assertEqual(lake.state_calls, 2)
        self.assertEqual(set(summary["tables"].keys()), {"income_vip", "fina_indicator_vip"})

    def test_refresh_range_facts_empty_table_warn_keeps_state_unmoved(self) -> None:
        client = _FakeClient()

        class _StubLake:
            def __init__(self) -> None:
                self.update_calls: list[dict[str, object]] = []

            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def update_ingestion_state(self, **kwargs):  # type: ignore[no-untyped-def]
                self.update_calls.append(dict(kwargs))
                return {}

        def _empty_balance(start_date: str, end_date: str) -> pd.DataFrame:
            _ = (start_date, end_date)
            return pd.DataFrame()

        client.fetch_balancesheet_vip = _empty_balance  # type: ignore[method-assign]
        lake = _StubLake()
        summary = self.mod._refresh_range_fact_tables(
            lake=lake,
            client=client,
            selected_range_fact_tables=["balancesheet_vip"],
            start_date="2026-01-01",
            end_date="2026-04-21",
            range_empty_policy="warn",
        )
        self.assertEqual(len(lake.update_calls), 0)
        self.assertFalse(bool(summary["tables"]["balancesheet_vip"]["state_updated"]))
        self.assertEqual(int(summary["tables"]["balancesheet_vip"]["raw_rows_filtered"]), 0)

    def test_refresh_range_facts_enforces_ann_date_boundary(self) -> None:
        client = _FakeClient()

        class _StubLake:
            def __init__(self) -> None:
                self.update_calls: list[dict[str, object]] = []

            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def update_ingestion_state(self, **kwargs):  # type: ignore[no-untyped-def]
                self.update_calls.append(dict(kwargs))
                return {}

        def _income_with_out_of_range(start_date: str, end_date: str) -> pd.DataFrame:
            _ = (start_date, end_date)
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "ann_date": ["20260410", "20260425"],
                    "end_date": ["20260331", "20260331"],
                    "total_revenue": [100.0, 999.0],
                    "revenue": [90.0, 900.0],
                    "n_income_attr_p": [10.0, 90.0],
                }
            )

        client.fetch_income_vip = _income_with_out_of_range  # type: ignore[method-assign]
        lake = _StubLake()
        summary = self.mod._refresh_range_fact_tables(
            lake=lake,
            client=client,
            selected_range_fact_tables=["income_vip"],
            start_date="2026-01-01",
            end_date="2026-04-21",
            enforce_ann_date_boundary=True,
        )
        self.assertEqual(len(lake.update_calls), 1)
        self.assertEqual(int(summary["tables"]["income_vip"]["raw_rows"]), 2)
        self.assertEqual(int(summary["tables"]["income_vip"]["raw_rows_filtered"]), 1)
        self.assertEqual(int(summary["tables"]["income_vip"]["ann_date_filter"]["dropped_rows"]), 1)

    def test_refresh_range_facts_flushes_each_window_without_full_concat(self) -> None:
        client = _FakeClient()

        class _StubLake:
            def __init__(self) -> None:
                self.vendor_rows: list[int] = []
                self.curated_rows: list[int] = []
                self.update_calls: list[dict[str, object]] = []

            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                self.vendor_rows.append(int(len(kwargs.get("df", pd.DataFrame()))))
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                self.curated_rows.append(int(len(kwargs.get("df", pd.DataFrame()))))
                return {}

            def update_ingestion_state(self, **kwargs):  # type: ignore[no-untyped-def]
                self.update_calls.append(dict(kwargs))
                return {}

        def _income_one_row(start_date: str, end_date: str) -> pd.DataFrame:
            client.calls["income_vip"] += 1
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "ann_date": [end_date],
                    "end_date": ["20260331"],
                    "total_revenue": [100.0],
                    "revenue": [90.0],
                    "n_income_attr_p": [10.0],
                }
            )

        def _no_full_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
            if len(frames) > 1:
                raise AssertionError("range facts should not concatenate all windows in memory")
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        client.fetch_income_vip = _income_one_row  # type: ignore[method-assign]
        original_concat = self.mod._concat_frames
        self.mod._concat_frames = _no_full_concat
        try:
            lake = _StubLake()
            summary = self.mod._refresh_range_fact_tables(
                lake=lake,
                client=client,
                selected_range_fact_tables=["income_vip"],
                start_date="2026-01-01",
                end_date="2026-04-21",
                range_window_days=30,
            )
        finally:
            self.mod._concat_frames = original_concat

        self.assertGreater(client.calls["income_vip"], 1)
        self.assertEqual(lake.vendor_rows, [1, 1, 1, 1])
        self.assertEqual(lake.curated_rows, [1, 1, 1, 1])
        self.assertEqual(len(lake.update_calls), 1)
        self.assertEqual(int(summary["tables"]["income_vip"]["raw_rows_filtered"]), 4)

    def test_flush_code_range_batch_releases_memory_after_write(self) -> None:
        class _StubLake:
            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

        releases: list[int] = []
        original_release = self.mod._release_process_memory
        self.mod._release_process_memory = lambda: releases.append(1)
        try:
            out = self.mod._flush_code_range_batch(
                lake=_StubLake(),
                table="cyq_perf",
                raw_frames=[
                    pd.DataFrame(
                        {
                            "ts_code": ["000001.SZ"],
                            "trade_date": ["2026-04-20"],
                            "winner_rate": [0.5],
                        }
                    )
                ],
                vendor_spec={
                    "table": "cyq_perf",
                    "date_col": "trade_date",
                    "key_cols": ("ts_code", "trade_date"),
                },
                curated_spec={
                    "table": "facts/cyq_perf",
                    "date_col": "date",
                    "key_cols": ("code", "date"),
                },
            )
        finally:
            self.mod._release_process_memory = original_release

        self.assertEqual(int(out["raw_rows"]), 1)
        self.assertGreaterEqual(len(releases), 2)


class TestBootstrapMemoryRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_bootstrap_module()

    def test_bootstrap_flush_trade_batch_releases_memory_after_writes(self) -> None:
        class _StubLake:
            def write_vendor_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def write_curated_trade_table(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

            def update_ingestion_state(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                return {}

        releases: list[int] = []
        original_release = self.mod._release_process_memory
        self.mod._release_process_memory = lambda: releases.append(1)
        try:
            out = self.mod._flush_trade_batch(
                lake=_StubLake(),
                adjust_mode="qfq",
                batch_dates=["20260420"],
                daily_frames=[],
                daily_basic_frames=[],
                adj_factor_frames=[],
                stk_limit_frames=[],
                suspend_d_frames=[],
                moneyflow_ths_frames=[],
            )
        finally:
            self.mod._release_process_memory = original_release

        self.assertEqual(int(out["batch_days"]), 1)
        self.assertGreaterEqual(len(releases), 2)


if __name__ == "__main__":
    unittest.main()
