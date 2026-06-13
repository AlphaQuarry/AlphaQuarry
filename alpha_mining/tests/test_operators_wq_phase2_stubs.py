from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 2 scoped stub: group_backfill is design-only this round", strict=False)
def test_group_backfill_design_stub() -> None:
    pytest.xfail("group_backfill implementation deferred with design+risk notes")


@pytest.mark.xfail(reason="Phase 2 scoped stub: ts_beta is design-only this round", strict=False)
def test_ts_beta_design_stub() -> None:
    pytest.xfail("ts_beta implementation deferred with design+risk notes")


@pytest.mark.xfail(reason="Phase 2 scoped stub: ts_resid is design-only this round", strict=False)
def test_ts_resid_design_stub() -> None:
    pytest.xfail("ts_resid implementation deferred with design+risk notes")
