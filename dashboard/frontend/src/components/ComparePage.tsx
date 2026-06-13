import { useEffect, useMemo, useState } from "react";
import { GitCompare, RefreshCw } from "lucide-react";

import { fetchRunCompare, fetchRuns } from "../api";
import type { AnalysisRun, RunCompareResponse, UniverseSummary } from "../types";
import { errorMessage, isAbortError } from "../utils/format";
import { STATUS_COPY } from "../utils/statusCopy";

export function ComparePage({
  universe,
  universes,
  onUniverseChange
}: {
  universe: string;
  universes: UniverseSummary[];
  onUniverseChange: (value: string) => void;
}) {
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [leftRunId, setLeftRunId] = useState("");
  const [rightRunId, setRightRunId] = useState("");
  const [topN, setTopN] = useState(50);
  const [compare, setCompare] = useState<RunCompareResponse | null>(null);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!universe) return;
    const controller = new AbortController();
    setError("");
    fetchRuns(universe, refreshKey > 0)
      .then((rows) => {
        const realRuns = rows.filter((run) => !run.is_scoreboard);
        setRuns(realRuns);
        const comparable = realRuns.filter((run) => run.factor_count > 0 && (run.has_dashboard_metrics || run.has_factor_metrics));
        const candidates = comparable.length >= 2 ? comparable : realRuns;
        setLeftRunId((current) => (candidates.some((run) => run.run_id === current) ? current : candidates[0]?.run_id ?? ""));
        setRightRunId((current) => {
          if (candidates.some((run) => run.run_id === current) && current !== (candidates[0]?.run_id ?? "")) return current;
          return candidates.find((run) => run.run_id !== (candidates[0]?.run_id ?? ""))?.run_id ?? candidates[0]?.run_id ?? "";
        });
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [universe, refreshKey]);

  useEffect(() => {
    if (!universe || !leftRunId || !rightRunId || leftRunId === rightRunId) {
      setCompare(null);
      return;
    }
    const controller = new AbortController();
    setError("");
    fetchRunCompare({ universe, leftRunId, rightRunId, topN, signal: controller.signal })
      .then(setCompare)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [universe, leftRunId, rightRunId, topN, refreshKey]);

  const comparableCount = runs.filter((run) => run.factor_count > 0 && (run.has_dashboard_metrics || run.has_factor_metrics)).length;
  const runOptions = useMemo(() => runs.map((run) => ({ id: run.run_id, label: `${run.run_id} - ${run.created_at_utc.slice(0, 10)}` })), [runs]);

  return (
    <section className="data-page compare-page">
      <header className="data-header">
        <div>
          <strong>Run Compare</strong>
          <span>Compare analysis runs by summary metrics and top-factor overlap</span>
        </div>
        <dl>
          <div><dt>Universe</dt><dd>{universe || "-"}</dd></div>
          <div><dt>Runs</dt><dd>{runs.length}</dd></div>
          <div><dt>Overlap</dt><dd>{compare ? `${Math.round(compare.overlap.overlap_ratio * 100)}%` : "-"}</dd></div>
        </dl>
      </header>

      {error ? <div className="data-error">{STATUS_COPY.error.api}: {error}</div> : null}

      <section className="field-panel">
        <div className="field-toolbar">
          <label>
            Universe
            <select value={universe} onChange={(event) => onUniverseChange(event.target.value)}>
              {universes.map((row) => <option key={row.name} value={row.name}>{row.name}</option>)}
            </select>
          </label>
          <label>
            Left run
            <select value={leftRunId} onChange={(event) => setLeftRunId(event.target.value)}>
              {runOptions.map((run) => <option key={run.id} value={run.id}>{run.label}</option>)}
            </select>
          </label>
          <label>
            Right run
            <select value={rightRunId} onChange={(event) => setRightRunId(event.target.value)}>
              {runOptions.map((run) => <option key={run.id} value={run.id}>{run.label}</option>)}
            </select>
          </label>
          <label>
            Top N
            <input type="number" min={1} max={500} value={topN} onChange={(event) => setTopN(Number(event.target.value) || 50)} />
          </label>
          <button type="button" className="compact-button" onClick={() => setRefreshKey((value) => value + 1)}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </section>

      {runs.length < 2 ? (
        <div className="metrics-empty">{STATUS_COPY.empty.runCompareNeedsTwoRuns}</div>
      ) : null}
      {runs.length >= 2 && comparableCount < 2 ? (
        <div className="metrics-empty">{STATUS_COPY.empty.runCompareNeedsMetrics}</div>
      ) : null}
      {leftRunId === rightRunId && runs.length >= 2 ? (
        <div className="metrics-empty">{STATUS_COPY.empty.chooseDifferentRuns}</div>
      ) : null}

      {compare ? (
        <section className="compare-grid">
          <section className="field-panel">
            <div className="superalpha-panel-title">
              <strong>Run Summary</strong>
              <span><GitCompare size={14} /> {compare.top_n} top factors</span>
            </div>
            <div className="field-table-wrap">
              <table className="field-table compact">
                <thead>
                  <tr>
                    <th>Property</th>
                    <th>Left</th>
                    <th>Right</th>
                  </tr>
                </thead>
                <tbody>
                  {["run_id", "created_at_utc", "period", "layers", "factor_count", "has_portfolio_pnl", "has_analysis_data"].map((key) => (
                    <tr key={key}>
                      <th>{label(key)}</th>
                      <td>{String((compare.left as unknown as Record<string, unknown>)[key] ?? "-")}</td>
                      <td>{String((compare.right as unknown as Record<string, unknown>)[key] ?? "-")}</td>
                    </tr>
                  ))}
                  <tr>
                    <th>Artifact Status</th>
                    <td>{artifactStatusLabel(compare.left_artifact_status)}</td>
                    <td>{artifactStatusLabel(compare.right_artifact_status)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            {compare.warnings.length ? (
              <div className="overview-message-list standalone">
                {compare.warnings.map((warning) => <p key={warning}>{warning}</p>)}
              </div>
            ) : null}
          </section>
          <section className="field-panel">
            <div className="superalpha-panel-title"><strong>Metric Delta</strong><span>right - left</span></div>
            <div className="field-table-wrap">
              <table className="field-table compact">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Left</th>
                    <th>Right</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(compare.metrics).map(([key, row]) => (
                    <tr key={key}>
                      <th>{label(key)}</th>
                      <td>{formatNumber(row.left)}</td>
                      <td>{formatNumber(row.right)}</td>
                      <td>{formatNumber(row.delta, true)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="field-panel compare-overlap">
            <div className="superalpha-panel-title">
              <strong>Top Factor Overlap</strong>
              <span>{compare.overlap.overlap_count} shared</span>
            </div>
            <div className="live-state-list">
              <div><span>Overlap ratio</span><strong>{Math.round(compare.overlap.overlap_ratio * 100)}%</strong></div>
              <div><span>Left only</span><strong>{compare.overlap.left_only.length}</strong></div>
              <div><span>Right only</span><strong>{compare.overlap.right_only.length}</strong></div>
            </div>
            <FactorList title="Shared" rows={compare.overlap.shared_factors} />
            <FactorList title="Left only" rows={compare.overlap.left_only} />
            <FactorList title="Right only" rows={compare.overlap.right_only} />
          </section>
        </section>
      ) : null}
    </section>
  );
}

function FactorList({ title, rows }: { title: string; rows: string[] }) {
  return (
    <div className="compare-factor-list">
      <strong>{title}</strong>
      <p>{rows.length ? `${rows.slice(0, 16).join(", ")}${rows.length > 16 ? ` ... (${rows.length} total)` : ` (${rows.length} total)`}` : "-"}</p>
    </div>
  );
}

function label(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatNumber(value: number | null | undefined, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const num = Number(value);
  const text = Math.abs(num) >= 10 ? num.toFixed(1) : num.toFixed(4);
  return signed && num > 0 ? `+${text}` : text;
}

function artifactStatusLabel(value?: string): string {
  if (value === "complete") return "Complete";
  if (value === "partial_metrics") return "Partial metrics";
  if (value === "missing_metrics") return STATUS_COPY.missing.artifact;
  if (value === "invalid_metrics") return "Invalid artifact";
  return value || "-";
}
