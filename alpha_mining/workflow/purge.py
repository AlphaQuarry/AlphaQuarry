from __future__ import annotations

from .lifecycle import mark_purged
from .universe_store import delete_universe_alpha_values


def purge_alpha_values(
    alpha_names: list[str],
    base_dir: str,
    universe_name: str,
    update_lifecycle: bool = True,
) -> dict[str, object]:
    result = delete_universe_alpha_values(
        alpha_names=alpha_names,
        base_dir=base_dir,
        universe_name=universe_name,
    )
    if update_lifecycle and alpha_names:
        mark_purged(
            alpha_names=[str(x) for x in alpha_names if str(x)],
            base_dir=base_dir,
            universe_name=universe_name,
        )
    return result
