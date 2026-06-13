import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

import { fetchLibrary } from "../api";
import type { LibraryResponse, UniverseSummary } from "../types";
import { errorMessage, formatNumber, isAbortError } from "../utils/format";

export function ResearchPage({
  universe,
  universes,
  onUniverseChange
}: {
  universe: string;
  universes: UniverseSummary[];
  onUniverseChange: (value: string) => void;
}) {
  const [payload, setPayload] = useState<LibraryResponse | null>(null);
  const [error, setError] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetchLibrary(universe, controller.signal)
      .then(setPayload)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      });
    return () => controller.abort();
  }, [universe, refreshKey]);

  const rows = (payload?.factors ?? []).filter((row) => {
    if (statusFilter !== "all" && String(row.status || "") !== statusFilter) {
      return false;
    }
    const q = query.trim().toLowerCase();
    if (!q) {
      return true;
    }
    return [row.factor, row.expression, row.analysis_run_id, row.nearest_factor_id, row.rejection_reason, row.library_status_reason]
      .map((value) => String(value || "").toLowerCase())
      .join(" ")
      .includes(q);
  });
  return (
    <section className="data-page">
      <header className="data-header">
        <div>
          <strong>Factor Library</strong>
          <span>{payload?.status === "ok" ? "Read-only accepted / staging / rejected registry" : payload?.message ?? "No library registry"}</span>
        </div>
        <dl>
          <div>
            <dt>Rows</dt>
            <dd>{rows.length.toLocaleString()} / {(payload?.total ?? 0).toLocaleString()}</dd>
          </div>
        </dl>
      </header>
      <section className="field-panel">
        <div className="field-toolbar">
          <label>
            Universe
            <select value={universe} onChange={(event) => onUniverseChange(event.target.value)}>
              {universes.map((row) => (
                <option key={row.name} value={row.name}>{row.name}</option>
              ))}
            </select>
          </label>
          <label>
            Status
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
              <option value="all">All</option>
              <option value="accepted">Accepted</option>
              <option value="staging">Staging</option>
              <option value="rejected">Rejected</option>
            </select>
          </label>
          <label>
            Search
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Factor, run, reason" />
          </label>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button type="button" className="compact-button" onClick={() => setRefreshKey((value) => value + 1)}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>
      </section>
      {error ? <div className="data-error">{error}</div> : null}
      <section className="field-panel">
        <div className="field-table-wrap">
          <table className="field-table">
            <thead>
              <tr>
                <th>Factor</th>
                <th>Status</th>
                <th>Mode</th>
                <th>Score</th>
                <th>Basis</th>
                <th>Submitted</th>
                <th>Run</th>
                <th>Signal Corr</th>
                <th>IC Corr</th>
                <th>Long-only Corr</th>
                <th>Long-short Corr</th>
                <th>Max PnL Corr</th>
                <th>Nearest</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {!rows.length ? <tr><td colSpan={14} className="empty-cell">No library factors</td></tr> : null}
              {rows.map((row) => (
                <tr key={`${row.universe ?? universe}-${row.analysis_run_id ?? ""}-${row.factor}`}>
                  <td className="field-name">{row.factor}</td>
                  <td>{row.library_status_effective || row.status || "-"}</td>
                  <td>{row.acceptance_mode || "-"}</td>
                  <td>{formatNumber(row.score ?? null, 1)}</td>
                  <td>{row.score_basis || "-"}</td>
                  <td>{row.submitted_at_utc || "-"}</td>
                  <td>{row.analysis_run_id || "-"}</td>
                  <td>{formatNumber(row.signal_corr ?? row.max_signal_corr ?? null, 3)}</td>
                  <td>{formatNumber(row.ic_corr ?? row.max_ic_corr ?? null, 3)}</td>
                  <td>{formatNumber(row.long_only_corr ?? null, 3)}</td>
                  <td>{formatNumber(row.long_short_corr ?? null, 3)}</td>
                  <td>{formatNumber(row.max_pnl_corr ?? null, 3)}</td>
                  <td>{row.nearest_factor_id || "-"}</td>
                  <td>{row.legacy_status_warning || row.rejection_reason || row.library_status_reason || row.reject_reasons || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
