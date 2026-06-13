from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def run_preflight_checks(
    *,
    config: str | Path = "configs/datasource.local.yaml",
    root: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config)
    base = Path(root) if root is not None else _infer_root(cfg_path)
    warnings: list[str] = []
    infos: list[str] = []
    remediations: list[str] = []

    gitignore = base / ".gitignore"
    if not gitignore.exists():
        warnings.append(".gitignore is missing")
        remediations.append("Add .gitignore entries for configs/datasource.local.yaml and .env.")
    else:
        text = gitignore.read_text(encoding="utf-8", errors="ignore")
        for required in ["configs/datasource.local.yaml", ".env"]:
            if required not in text:
                warnings.append(f".gitignore missing required entry: {required}")
                remediations.append(f"Add {required} to .gitignore before storing local secrets.")

    if cfg_path.exists():
        try:
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            warnings.append(f"failed to parse local datasource config: {exc}")
            payload = {}
        if isinstance(payload, dict):
            tsh = payload.get("tushare", {})
            if isinstance(tsh, dict):
                token = str(tsh.get("token", "") or "").strip()
                http_url = str(tsh.get("http_url", "") or "").strip()
                if token:
                    warnings.append(
                        "local datasource config contains non-empty tushare.token; "
                        "use environment variable TUSHARE_TOKEN instead"
                    )
                    remediations.append(
                        "Clear tushare.token in the local config and set TUSHARE_TOKEN in the shell environment."
                    )
                if http_url:
                    infos.append(
                        "local datasource config has custom tushare.http_url configured; ensure this endpoint is private and expected"
                    )
    else:
        infos.append("local datasource config does not exist; this is ok for a fresh setup")

    status = "ok" if not warnings else "warn"
    return {
        "status": status,
        "warnings": warnings,
        "infos": infos,
        "remediations": sorted(set(remediations)),
        "strict_exit_code": 0 if not warnings else 2,
    }


def _infer_root(config: Path) -> Path:
    if config.is_absolute() and config.parent.name == "configs":
        return config.parent.parent
    return Path.cwd()
