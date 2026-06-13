from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.mining.operator_signatures import (
    build_default_operator_signature_registry,
)
from alpha_mining.registry import build_default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit expression operator implementations and signatures.")
    parser.add_argument("--output", default="artifacts/dev/operator_signature_audit.csv")
    args = parser.parse_args()

    impl = set(build_default_registry().list_names())
    sig_registry = build_default_operator_signature_registry()
    sig = set(sig_registry.names())
    missing_signature = sorted(impl - sig)
    missing_implementation = sorted(sig - impl)

    rows = []
    for name in sorted(impl | sig):
        specs = sig_registry.get_all(name)
        rows.append(
            {
                "operator": name,
                "implemented": name in impl,
                "signed": name in sig,
                "signature_count": len(specs),
                "signatures": ";".join(f"({','.join(spec.input_types)})->{spec.output_type}" for spec in specs),
                "status": (
                    "missing_signature"
                    if name in missing_signature
                    else "missing_implementation"
                    if name in missing_implementation
                    else "ok"
                ),
            }
        )

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(f"[operator-audit] implemented={len(impl)} signed={len(sig)}")
    print(f"[operator-audit] missing_signature={missing_signature}")
    print(f"[operator-audit] missing_implementation={missing_implementation}")
    print(f"[operator-audit] output={output.as_posix()}")
    if missing_signature or missing_implementation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
