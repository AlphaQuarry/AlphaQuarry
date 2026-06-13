from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass(frozen=True)
class BenchmarkBinding:
    code: str
    source: str
    universe_name: str
    reason: str = ""


INDEX_BENCHMARK_BY_UNIVERSE: dict[str, str] = {
    "hs300": "000300.SH",
    "csi500": "000905.SH",
    "csi1000": "000852.SH",
    "csi2000": "932000.CSI",
    "csi_all_share": "000985.SH",
    "cnindex2000": "399303.SZ",
    "sme_composite": "399101.SZ",
    "cn_all": "000985.SH",
}


def resolve_benchmark_binding(*, universe_name: str, explicit_code: str = "") -> BenchmarkBinding:
    universe = str(universe_name or "").strip()
    code = str(explicit_code or "").strip()
    if code:
        return BenchmarkBinding(code=code, source="explicit", universe_name=universe)
    key = _normalize_universe_key(universe)
    bound = INDEX_BENCHMARK_BY_UNIVERSE.get(key)
    if bound:
        return BenchmarkBinding(code=bound, source="universe", universe_name=universe)
    logging.getLogger("alpha_mining.benchmark_binding").warning(
        "[benchmark_binding] universe=%s is not mapped; fallback benchmark=000300.SH",
        universe,
    )
    return BenchmarkBinding(
        code="000300.SH",
        source="fallback",
        universe_name=universe,
        reason="unmapped_universe",
    )


def _normalize_universe_key(universe_name: str) -> str:
    text = str(universe_name or "").strip().lower()
    prefix = "cn_all_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text
