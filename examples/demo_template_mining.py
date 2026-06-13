from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.mining import load_templates
from alpha_mining.mining.pipeline import AlphaMiningPipeline
from alpha_mining.panel_store import PanelStore


def main() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "code": ["A", "B", "A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 19.0, 12.0, 21.0],
            "volume": [100, 200, 110, 210, 120, 220],
            "industry": ["I1", "I2", "I1", "I2", "I1", "I2"],
            "in_universe": [1, 1, 1, 0, 1, 1],
        }
    )
    store = PanelStore.from_long_frame(df, group_fields=["industry"])
    config = AlphaMiningConfig(
        simulation=AlphaSimulationConfig(delay=0, neutralization="NONE", universe="in_universe")
    )
    pipeline = AlphaMiningPipeline.from_panel_store(store, config=config)
    templates = load_templates(include_families=set(config.prioritized_template_families))
    alpha_df, failed = pipeline.run_templates(templates=templates)
    print(alpha_df.head())
    print("failed:", failed)


if __name__ == "__main__":
    main()
