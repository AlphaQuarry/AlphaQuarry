from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.workflow.reproduce import (
    reproduce_alpha_by_expression,
    reproduce_alpha_by_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce alpha values by alpha_name or expression.")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--base-dir", default="data/alpha_universe_store")
    parser.add_argument("--alpha-name", default="")
    parser.add_argument("--expression", default="")
    parser.add_argument("--manifest-id", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()

    if bool(args.alpha_name) == bool(args.expression):
        raise ValueError("Specify exactly one of --alpha-name or --expression")

    output_stem = Path(args.output) if args.output else None
    if args.alpha_name:
        out = reproduce_alpha_by_name(
            alpha_name=str(args.alpha_name),
            base_dir=str(args.base_dir),
            universe_name=str(args.universe),
            manifest_id=str(args.manifest_id or "") or None,
            compare_with_saved=not bool(args.no_compare),
            save_path_stem=output_stem,
        )
    else:
        out = reproduce_alpha_by_expression(
            expression=str(args.expression),
            base_dir=str(args.base_dir),
            universe_name=str(args.universe),
            manifest_id=str(args.manifest_id or "") or None,
            save_path_stem=output_stem,
        )

    df = out.get("output_df")
    if isinstance(df, pd.DataFrame):
        print("reproduced rows =", len(df))
        print(df.head(10))
    print("meta =", {k: v for k, v in out.items() if k != "output_df"})


if __name__ == "__main__":
    main()
