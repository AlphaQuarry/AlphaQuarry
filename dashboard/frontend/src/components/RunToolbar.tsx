import { RefreshCw, Search, Star } from "lucide-react";

import type { AnalysisRun, UniverseSummary } from "../types";
import { formatRunLabel } from "../utils/format";

type RunToolbarProps = {
  universes: UniverseSummary[];
  selectedUniverse: string;
  runs: AnalysisRun[];
  selectedRunId: string;
  query: string;
  effectiveOnly: boolean;
  onUniverseChange: (value: string) => void;
  onRunChange: (value: string) => void;
  onQueryChange: (value: string) => void;
  onEffectiveOnlyChange: () => void;
  onRefresh: () => void;
};

export function RunToolbar({
  universes,
  selectedUniverse,
  runs,
  selectedRunId,
  query,
  effectiveOnly,
  onUniverseChange,
  onRunChange,
  onQueryChange,
  onEffectiveOnlyChange,
  onRefresh
}: RunToolbarProps) {
  return (
    <section className="toolbar-band">
      <div className="toolbar-left">
        <label>
          Universe
          <select value={selectedUniverse} onChange={(event) => onUniverseChange(event.target.value)}>
            {universes.map((row) => (
              <option key={row.name} value={row.name}>
                {row.name} ({row.run_count})
              </option>
            ))}
          </select>
        </label>
        <label>
          Run
          <select value={selectedRunId} onChange={(event) => onRunChange(event.target.value)}>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {formatRunLabel(run)}
              </option>
            ))}
          </select>
        </label>
        <div className="search-box">
          <Search size={16} />
          <input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder="Search factor or expression" />
        </div>
      </div>
      <div className="toolbar-right">
        <button className={effectiveOnly ? "toggle active" : "toggle"} type="button" onClick={onEffectiveOnlyChange}>
          <Star size={15} />
          Effective
        </button>
        <button className="icon-button" type="button" title="Refresh" onClick={onRefresh}>
          <RefreshCw size={17} />
        </button>
      </div>
    </section>
  );
}
