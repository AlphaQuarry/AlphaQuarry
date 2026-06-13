from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
import uuid

from alpha_mining.datasource.config import load_datasource_settings


class TestDatasourceConfigEnvOverride(unittest.TestCase):
    def test_missing_env_does_not_override_yaml_defaults(self) -> None:
        cfg_path = self._write_temp_config()
        with patch.dict(os.environ, {}, clear=True):
            settings = load_datasource_settings(cfg_path)
        self.assertEqual(settings.universe_min_days_since_listed, 60)
        self.assertEqual(settings.update_lookback_trade_days, 5)
        self.assertEqual(settings.tushare.max_retries, 3)
        self.assertAlmostEqual(settings.tushare.retry_sleep_seconds, 1.5)

    def test_env_can_override_yaml_values(self) -> None:
        cfg_path = self._write_temp_config()
        with patch.dict(
            os.environ,
            {
                "PROJECT_UNIVERSE_MIN_DAYS_SINCE_LISTED": "90",
                "PROJECT_UPDATE_LOOKBACK_TRADE_DAYS": "7",
                "TUSHARE_MAX_RETRIES": "4",
                "TUSHARE_RETRY_SLEEP_SECONDS": "2.0",
            },
            clear=True,
        ):
            settings = load_datasource_settings(cfg_path)
        self.assertEqual(settings.universe_min_days_since_listed, 90)
        self.assertEqual(settings.update_lookback_trade_days, 7)
        self.assertEqual(settings.tushare.max_retries, 4)
        self.assertAlmostEqual(settings.tushare.retry_sleep_seconds, 2.0)

    def _write_temp_config(self) -> str:
        temp_dir = Path("data") / f"_tmp_cfg_test_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        path = temp_dir / "datasource.yaml"
        path.write_text(
            "\n".join(
                [
                    "source_backend: duckdb",
                    "source_view: v_project_panel_cn_a",
                    "adjust_mode: qfq",
                    "universe:",
                    "  min_days_since_listed: 60",
                    "  exclude_st: true",
                    "  include_bj: true",
                    "update:",
                    "  lookback_trade_days: 5",
                    "tushare:",
                    "  max_retries: 3",
                    "  retry_sleep_seconds: 1.5",
                ]
            ),
            encoding="utf-8",
        )
        return str(path.as_posix())


if __name__ == "__main__":
    unittest.main()
