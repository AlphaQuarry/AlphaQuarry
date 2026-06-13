"""Tushare 权限预检脚本。

通过 3 次代表性 API 调用，快速判定用户积分档（2000/5000/10000），
推断全部接口可用性，避免逐一测试 17+ 个接口。

用法:
    python scripts/preflight_tushare.py [--config configs/datasource.local.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import (
    TushareApiError,
    TushareErrorCategory,
    build_tushare_client_from_settings,
    load_datasource_settings,
)
from alpha_mining.datasource.ingestion_scope import TIER_PRESETS
from alpha_mining.datasource.tushare_client import classify_tushare_error

# 积分档阈值及其代表接口
# 每档测试一个接口即可判定用户是否达到该积分
_TIER_PROBES: list[tuple[int, str, dict, str]] = [
    # (积分档, 接口名, 测试参数, 说明)
    (2000, "daily", {"trade_date": "20240102"}, "P0/P1/P2/P3/index 核心数据"),
    (5000, "stk_auction_o", {"trade_date": "20240102"}, "P2(VIP财务)/P4_auction"),
    (10000, "cyq_perf", {"trade_date": "20240102"}, "P4(筹码/技术因子)/P5(券商预测)"),
]

# 全部接口的积分要求 (基于 Tushare 官方文档)
_TABLE_POINTS: dict[str, int] = {
    # P0 dim (2000)
    "stock_basic": 2000,
    "trade_cal": 2000,
    "index_classify": 2000,
    "index_member_all": 2000,
    # P1 dim (2000)
    "namechange": 2000,
    # P0 fact (2000)
    "daily": 2000,
    "daily_basic": 2000,
    "adj_factor": 2000,
    # P1 fact (2000)
    "stk_limit": 2000,
    "suspend_d": 2000,
    # P2 fact (5000, VIP版全市场拉取)
    "income_vip": 5000,
    "balancesheet_vip": 5000,
    "cashflow_vip": 5000,
    "fina_indicator_vip": 5000,
    # P3 fact (2000)
    "moneyflow": 2000,
    "moneyflow_ths": 2000,
    # P4 fact (10000, 特色数据)
    "cyq_perf": 10000,
    "cyq_chips": 10000,
    "stk_factor_pro": 10000,
    # P4_auction (5000)
    "stk_auction_o": 5000,
    "stk_auction_c": 5000,
    # P5 (10000, 特色数据)
    "report_rc": 10000,
    # index (2000)
    "index_daily": 2000,
    "index_weight": 2000,
}

# 积分档到数据组的映射
_TIER_GROUPS: dict[int, list[str]] = {
    2000: ["P0", "P1", "P2(非VIP)", "P3", "index"],
    5000: ["P2(VIP财务)", "P4_auction(集合竞价)"],
    10000: ["P4(筹码/技术因子)", "P5(券商预测)"],
}


def _test_probe(client, api_name: str, params: dict) -> tuple[str, str]:
    """测试单个探测接口。返回 (状态, 原因)。"""
    try:
        fn = getattr(client._pro, api_name)
        result = fn(**params)
        if result is None or (hasattr(result, "empty") and result.empty):
            return ("ok", "可用")
        return ("ok", f"可用 ({len(result)} 行)")
    except TushareApiError as exc:
        if exc.category == TushareErrorCategory.AUTH:
            return ("denied", "权限不足")
        if exc.category == TushareErrorCategory.RATE_LIMIT:
            return ("rate_limited", "限频 (可用但慢)")
        return ("error", f"[{exc.category.value}] {str(exc)[:80]}")
    except Exception as exc:
        cat = classify_tushare_error(exc)
        if cat == TushareErrorCategory.AUTH:
            return ("denied", "权限不足")
        if cat == TushareErrorCategory.RATE_LIMIT:
            return ("rate_limited", "限频 (可用但慢)")
        return ("error", f"[{cat.value}] {str(exc)[:80]}")


def run_preflight(config_path: str | None) -> None:
    settings = load_datasource_settings(config_path)
    client = build_tushare_client_from_settings(settings.tushare)

    print("[preflight] Tushare 权限预检")
    print("=" * 60)

    # 阶段 1: 探测积分档
    detected_points = 0
    probe_results: list[tuple[int, str, str, str]] = []

    for points, api_name, params, desc in _TIER_PROBES:
        status, reason = _test_probe(client, api_name, params)
        probe_results.append((points, api_name, status, reason))
        if status in ("ok", "rate_limited"):
            detected_points = max(detected_points, points)
        # 测试间隔
        time.sleep(2)

    print(f"\n📊 探测结果 ({len(_TIER_PROBES)} 次 API 调用):")
    for points, api_name, status, reason in probe_results:
        icon = {"ok": "✅", "rate_limited": "⏳", "denied": "❌", "error": "⚠️"}.get(status, "?")
        print(f"  {icon} {api_name:<25} {points:>5} 积分  {reason}")

    # 阶段 2: 推断全部接口可用性
    print(f"\n{'=' * 60}")
    print(f"判定积分档: ≥{detected_points}")

    available_tables: list[tuple[str, int]] = []
    denied_tables: list[tuple[str, int]] = []

    for table, required in sorted(_TABLE_POINTS.items(), key=lambda x: (x[1], x[0])):
        if required <= detected_points:
            available_tables.append((table, required))
        else:
            denied_tables.append((table, required))

    print(f"\n✅ 可用接口 ({len(available_tables)} 个):")
    for table, pts in available_tables:
        print(f"  {table:<25} {pts} 积分")

    if denied_tables:
        print(f"\n❌ 不可用接口 ({len(denied_tables)} 个):")
        for table, pts in denied_tables:
            print(f"  {table:<25} {pts} 积分")

    # 阶段 3: 建议 tier
    print(f"\n{'=' * 60}")
    recommended_tier = "basic"
    for tier_name in ("full", "extended", "standard", "basic"):
        _, _, tier_points = TIER_PRESETS[tier_name]
        if detected_points >= tier_points:
            recommended_tier = tier_name
            break

    tier_fact, tier_dim, tier_pts = TIER_PRESETS[recommended_tier]
    print(f"建议: --tier {recommended_tier}")
    print(f"  包含数据组: fact={list(tier_fact)}, dim={list(tier_dim)}")
    print(f"  推荐积分: {tier_pts}")

    # 列出所有可用 tier
    print("\n可用 tier 列表:")
    for tier_name in ("basic", "standard", "extended", "full"):
        _, _, tier_pts = TIER_PRESETS[tier_name]
        marker = " ← 推荐" if tier_name == recommended_tier else ""
        ok = "✅" if detected_points >= tier_pts else "❌"
        print(f"  {ok} --tier {tier_name:<10} 需要 {tier_pts} 积分{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tushare 权限预检")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    args = parser.parse_args()
    run_preflight(str(args.config or "") or None)


if __name__ == "__main__":
    main()
