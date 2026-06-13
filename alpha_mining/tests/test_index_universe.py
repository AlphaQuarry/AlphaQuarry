from __future__ import annotations

import pandas as pd
import pytest

from alpha_mining.datasource.index_universe import (
    load_index_universe_config,
    normalize_index_code,
    resolve_index_universes,
)


def test_load_default_index_universe_config_contains_supported_universes() -> None:
    specs = load_index_universe_config("configs/index_universes.yaml")

    assert list(specs) == [
        "hs300",
        "csi500",
        "csi1000",
        "csi2000",
        "csi_all_share",
        "cnindex2000",
        "sme_composite",
    ]
    assert specs["hs300"].candidate_codes[0] == "000300.SH"
    assert specs["cnindex2000"].required is False


def test_normalize_index_code_supports_exchange_aliases() -> None:
    assert normalize_index_code("399303.xshe") == "399303.SZ"
    assert normalize_index_code("000300.xshg") == "000300.SH"


def test_resolve_index_universes_warns_and_skips_missing_optional() -> None:
    specs = load_index_universe_config("configs/index_universes.yaml")
    index_basic = pd.DataFrame(
        {
            "code": ["000300.SH", "000905.SH"],
            "name": ["沪深300", "中证500"],
            "market": ["CSI", "CSI"],
            "publisher": ["中证公司", "中证公司"],
            "category": ["规模指数", "规模指数"],
        }
    )

    with pytest.warns(UserWarning, match="cnindex2000"):
        resolved = resolve_index_universes(
            specs,
            index_basic,
            universe_names=["hs300", "cnindex2000"],
            missing_policy="warn",
            snapshot_date="2026-05-21",
        )

    assert resolved.loc[resolved["universe_name"] == "hs300", "status"].iloc[0] == "active"
    assert resolved.loc[resolved["universe_name"] == "hs300", "index_weight_code"].iloc[0] == "000300.SH"
    assert resolved.loc[resolved["universe_name"] == "cnindex2000", "status"].iloc[0] == "missing"
