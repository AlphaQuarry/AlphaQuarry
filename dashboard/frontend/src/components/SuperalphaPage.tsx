import { useEffect, useMemo, useState } from "react";
import { BarChart3, Check, ChevronDown, ChevronUp, Images, LineChart, Pencil, RefreshCw, Search, X } from "lucide-react";

import {
  activateLiveSuperalpha,
  fetchActiveLiveSuperalphas,
  fetchSuperalphaComponents,
  fetchSuperalphaDetail,
  fetchSuperalphaRuns,
  renameSuperalphaRun,
  runSuperalphaBacktest
} from "../api";
import type { LiveSuperalpha, SuperalphaComponent, SuperalphaDetailResponse, SuperalphaRun, UniverseSummary } from "../types";
import { errorMessage, formatNumber, isAbortError } from "../utils/format";
import { AnalysisDataPanel } from "./AnalysisDataPanel";
import { MetricSummary } from "./MetricSummary";
import { PnlChart } from "./PnlChart";

type ResultTab = "pnl" | "analysis" | "metrics";

function shortSuperalphaId(superalphaId: string) {
  return `SA ${superalphaId.replace(/^superalpha_/, "").slice(0, 8) || superalphaId.slice(0, 8)}`;
}

function compactCombo(value?: string | null) {
  const text = String(value || "1").trim() || "1";
  return text.length > 34 ? `${text.slice(0, 31)}...` : text;
}

function summaryText(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (Array.isArray(value)) {
    return value.length ? `${value.length} item(s)` : "none";
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return "-";
    return entries.slice(0, 3).map(([key, item]) => `${key}: ${summaryText(item)}`).join(" | ");
  }
  return String(value);
}

export function SuperalphaPage({
  universe,
  universes,
  onUniverseChange
}: {
  universe: string;
  universes: UniverseSummary[];
  onUniverseChange: (value: string) => void;
}) {
  const [components, setComponents] = useState<SuperalphaComponent[]>([]);
  const [runs, setRuns] = useState<SuperalphaRun[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [comboExpression, setComboExpression] = useState("1");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [detail, setDetail] = useState<SuperalphaDetailResponse | null>(null);
  const [resultTab, setResultTab] = useState<ResultTab>("pnl");
  const [showTestPhase, setShowTestPhase] = useState(false);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [editingRunId, setEditingRunId] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const [renamingRunId, setRenamingRunId] = useState("");
  const [liveRows, setLiveRows] = useState<LiveSuperalpha[]>([]);
  const [activatingRunId, setActivatingRunId] = useState("");
  // Safety settings for SA runtime.
  const [componentJoin, setComponentJoin] = useState<"concat" | "inner">("concat");
  const [allowReproduceFallback, setAllowReproduceFallback] = useState(true);
  const [maxComponents, setMaxComponents] = useState(20);

  useEffect(() => {
    setSelectedRunId("");
    setDetail(null);
    setResultTab("pnl");
    setShowTestPhase(false);
    setSelectedIds([]);
    setEditingRunId("");
    setRenameValue("");
    setError("");
  }, [universe]);

  useEffect(() => {
    if (!universe) {
      setComponents([]);
      setRuns([]);
      setSelectedRunId("");
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    Promise.all([
      fetchSuperalphaComponents(universe, controller.signal),
      fetchSuperalphaRuns(universe, controller.signal),
      fetchActiveLiveSuperalphas(universe, controller.signal)
    ])
      .then(([componentPayload, runPayload, livePayload]) => {
        const nextRuns = runPayload.runs ?? [];
        setComponents(componentPayload.components ?? []);
        setRuns(nextRuns);
        setLiveRows(livePayload.superalphas ?? []);
        setSelectedIds((current) =>
          current.filter((id) => (componentPayload.components ?? []).some((row) => row.factor === id && row.can_backtest !== false))
        );
        setSelectedRunId((current) => {
          if (!current || nextRuns.some((row) => row.superalpha_id === current)) {
            return current;
          }
          setDetail(null);
          return "";
        });
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [universe, refreshKey]);

  useEffect(() => {
    if (!selectedRunId) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    const requestRunId = selectedRunId;
    const requestUniverse = universe;
    setDetailLoading(true);
    setError("");
    fetchSuperalphaDetail(requestRunId, controller.signal, showTestPhase)
      .then((payload) => {
        if (payload.superalpha_id === requestRunId && payload.universe === requestUniverse) {
          setDetail(payload);
        }
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => setDetailLoading(false));
    return () => controller.abort();
  }, [selectedRunId, showTestPhase, universe]);

  const filteredComponents = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return components;
    return components.filter((row) =>
      [row.factor, row.expression, row.acceptance_mode, row.analysis_run_id]
        .map((value) => String(value || "").toLowerCase())
        .join(" ")
        .includes(q)
    );
  }, [components, query]);

  const getSignalStatusBadge = (row: SuperalphaComponent) => {
    const status = row.signal_status;
    switch (status) {
      case "compact": return { label: "Compact", className: "badge-ok" };
      case "raw": return { label: "Raw", className: "badge-ok" };
      case "cached": return { label: "Cached", className: "badge-ok" };
      case "reproducible": return { label: "Reproducible", className: "badge-warn" };
      case "reproduced": return { label: "Reproduced", className: "badge-ok" };
      case "duckdb_fallback": return { label: "DuckDB Fallback", className: "badge-warn" };
      case "read_error": return { label: "Read Error", className: "badge-error" };
      case "unavailable": return { label: "Unavailable", className: "badge-error" };
      default: return row.signal_available === false
        ? { label: "Unavailable", className: "badge-error" }
        : { label: "Compact", className: "badge-ok" };
    }
  };
  const selectedOrder = useMemo(
    () => selectedIds.map((id) => components.find((row) => row.factor === id)).filter(Boolean) as SuperalphaComponent[],
    [components, selectedIds]
  );

  const selectedRun = useMemo(
    () => runs.find((run) => run.superalpha_id === selectedRunId) ?? null,
    [runs, selectedRunId]
  );

  const selectedRunComponents = useMemo(() => {
    if (selectedRun?.components?.length) {
      return selectedRun.components;
    }
    const metaComponents = detail?.meta && Array.isArray(detail.meta.components) ? detail.meta.components : [];
    return metaComponents as SuperalphaComponent[];
  }, [detail?.meta, selectedRun]);

  // Risk statistics for selected components
  const selectedRisk = useMemo(() => {
    const compact = selectedOrder.filter((r) => r.signal_status === "compact" || r.signal_status === "raw" || r.signal_status === "cached").length;
    const reproducible = selectedOrder.filter((r) => r.signal_status === "reproducible" || r.signal_status === "duckdb_fallback").length;
    const unavailable = selectedOrder.filter((r) => r.signal_status === "unavailable" || r.signal_status === "read_error" || r.can_backtest === false).length;
    return { compact, reproducible, unavailable, total: selectedOrder.length };
  }, [selectedOrder]);

  const canRun = selectedIds.length > 0
    && !running
    && selectedIds.length <= maxComponents
    && selectedRisk.unavailable === 0
    && (allowReproduceFallback || selectedRisk.reproducible === 0);

  const runBacktest = () => {
    if (!universe || !selectedIds.length || running) {
      return;
    }
    setRunning(true);
    setError("");
    runSuperalphaBacktest({
      universe,
      factor_ids: selectedIds,
      combo_expression: comboExpression || "1",
      component_join: componentJoin,
      allow_reproduce_fallback: allowReproduceFallback,
      max_components: maxComponents,
    })
      .then((payload) => {
        setSelectedRunId(payload.superalpha_id);
        setRefreshKey((value) => value + 1);
        setResultTab("pnl");
      })
      .catch((exc: unknown) => setError(errorMessage(exc)))
      .finally(() => setRunning(false));
  };

  const beginRename = (run: SuperalphaRun) => {
    setEditingRunId(run.superalpha_id);
    setRenameValue((run.display_name || run.name || "").trim());
    setError("");
  };

  const cancelRename = () => {
    setEditingRunId("");
    setRenameValue("");
  };

  const saveRename = (run: SuperalphaRun) => {
    const nextName = renameValue.trim();
    if (!nextName || renamingRunId) {
      return;
    }
    setRenamingRunId(run.superalpha_id);
    setError("");
    renameSuperalphaRun(run.superalpha_id, nextName)
      .then((payload) => {
        const updated = payload.run;
        setRuns((current) => current.map((item) => (item.superalpha_id === updated.superalpha_id ? { ...item, ...updated } : item)));
        setSelectedRunId(updated.superalpha_id);
        setEditingRunId("");
        setRenameValue("");
      })
      .catch((exc: unknown) => setError(errorMessage(exc)))
      .finally(() => setRenamingRunId(""));
  };

  const activateRunForLive = (run: SuperalphaRun) => {
    if (!universe || activatingRunId) return;
    setActivatingRunId(run.superalpha_id);
    setError("");
    activateLiveSuperalpha(universe, run.superalpha_id)
      .then(() => setRefreshKey((value) => value + 1))
      .catch((exc: unknown) => setError(errorMessage(exc)))
      .finally(() => setActivatingRunId(""));
  };

  const liveStatusFor = (superalphaId: string) => (
    liveRows.find((row) => row.superalpha_id === superalphaId)?.status || "not_active"
  );

  const toggleSelection = (factor: string, checked: boolean) => {
    setSelectedIds((current) => {
      if (checked) {
        return current.includes(factor) ? current : [...current, factor];
      }
      return current.filter((id) => id !== factor);
    });
  };

  const moveSelectedFactor = (factor: string, direction: -1 | 1) => {
    setSelectedIds((current) => {
      const index = current.indexOf(factor);
      if (index < 0) return current;
      const target = index + direction;
      if (target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  };

  return (
    <section className="data-page superalpha-page">
      <header className="data-header">
        <div>
          <strong>Superalpha</strong>
          <span>Manual accepted-factor combination and backtest</span>
        </div>
        <dl>
          <div>
            <dt>Components</dt>
            <dd>{components.length.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Selected</dt>
            <dd>{selectedIds.length.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Runs</dt>
            <dd>{runs.length.toLocaleString()}</dd>
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
          <label className="field-search">
            <Search size={15} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search accepted factors" />
          </label>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button type="button" className="compact-button" onClick={() => setRefreshKey((value) => value + 1)}>
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>
      </section>

      {error ? <div className="data-error">{error}</div> : null}

      <section className="superalpha-layout">
        <div className="field-panel">
          <div className="superalpha-panel-title">
            <strong>Accepted Factors</strong>
            <span>{loading ? "Loading..." : `${filteredComponents.length} visible`}</span>
          </div>
          <div className="field-table-wrap superalpha-table-wrap">
            <table className="field-table superalpha-component-table">
              <thead>
                <tr>
                  <th>Use</th>
                  <th>Factor</th>
                  <th>Score</th>
                  <th>Status</th>
                  <th>Mode</th>
                  <th>LO Sharpe</th>
                  <th>LS Sharpe</th>
                  <th>Submitted</th>
                  <th>Expression</th>
                </tr>
              </thead>
              <tbody>
                {!filteredComponents.length ? (
                  <tr><td colSpan={9} className="empty-cell">No accepted factors in this universe</td></tr>
                ) : null}
                {filteredComponents.map((row) => {
                  const badge = getSignalStatusBadge(row);
                  const isDisabled = row.can_backtest === false;
                  return (
                    <tr key={row.factor} className={isDisabled ? "unavailable-row" : undefined}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(row.factor)}
                          disabled={isDisabled}
                          onChange={(event) => toggleSelection(row.factor, event.target.checked)}
                        />
                      </td>
                      <td className="field-name">{row.factor}</td>
                      <td>{formatNumber(row.score ?? null, 1)}</td>
                      <td><span className={`signal-badge ${badge.className}`} title={row.reproduce_warning || row.signal_status_reason || ""}>{badge.label}</span></td>
                      <td>{row.acceptance_mode || "-"}</td>
                      <td>{formatNumber(row.candidate_long_only_sharpe ?? null, 2)}</td>
                      <td>{formatNumber(row.candidate_long_short_sharpe ?? null, 2)}</td>
                      <td>{row.submitted_at_utc || "-"}</td>
                      <td>{isDisabled ? row.signal_status_reason || "signal unavailable" : row.expression || "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="field-panel superalpha-control-panel">
          {/* Group 1: Combo Expression */}
          <div className="sa-ctrl-section">
            <div className="sa-ctrl-section-label">Combo Expression</div>
            <textarea
              value={comboExpression}
              onChange={(event) => setComboExpression(event.target.value)}
              rows={3}
              placeholder="1 = equal weight"
            />
            <div className="sa-ctrl-hint">Use [0.5,0.3,0.2] for custom weights following the selected order.</div>
          </div>

          {/* Group 2: Selected Order */}
          <div className="sa-ctrl-section">
            <div className="sa-ctrl-section-label">Selected Order</div>
            <div className="selected-order">
              {selectedOrder.length ? (
                selectedOrder.map((row, index) => (
                  <div key={row.factor} className="selected-order-item">
                    <span className="selected-order-name">{index + 1}. {row.factor}</span>
                    <span className="selected-order-arrows">
                      <button
                        type="button"
                        className="selected-order-arrow"
                        disabled={index === 0}
                        onClick={() => moveSelectedFactor(row.factor, -1)}
                        title="Move up"
                      >
                        <ChevronUp size={12} />
                      </button>
                      <button
                        type="button"
                        className="selected-order-arrow"
                        disabled={index === selectedOrder.length - 1}
                        onClick={() => moveSelectedFactor(row.factor, 1)}
                        title="Move down"
                      >
                        <ChevronDown size={12} />
                      </button>
                    </span>
                  </div>
                ))
              ) : (
                <span className="sa-ctrl-empty">No factor selected</span>
              )}
            </div>
          </div>

          {/* Group 3: Parameters */}
          <div className="sa-ctrl-section">
            <div className="sa-ctrl-section-label">Parameters</div>
            <div className="sa-param-row">
              <label>
                Join Mode
                <select value={componentJoin} onChange={(e) => setComponentJoin(e.target.value as "concat" | "inner")}>
                  <option value="concat">Concat (safe)</option>
                  <option value="inner">Inner overlap</option>
                </select>
              </label>
              <label>
                Max Components
                <select value={maxComponents} onChange={(e) => setMaxComponents(Number(e.target.value))}>
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={30}>30</option>
                  <option value={50}>50</option>
                </select>
              </label>
            </div>
            <label className="sa-fallback-toggle">
              <input type="checkbox" checked={allowReproduceFallback} onChange={(e) => setAllowReproduceFallback(e.target.checked)} />
              Allow reproduce fallback
            </label>
            {allowReproduceFallback ? (
              <div className="sa-callout sa-callout-danger">
                Reproduce fallback loads full data from DuckDB and may use significant memory.
              </div>
            ) : null}
          </div>

          {/* Group 4: Risk & Action */}
          <div className="sa-ctrl-section sa-risk-action">
            {selectedRisk.total > 0 ? (
              <div className="sa-risk-summary">
                <span className="sa-risk-ready">{selectedRisk.compact} ready</span>
                {selectedRisk.reproducible > 0 ? <span className="sa-risk-warn">{selectedRisk.reproducible} need reproduce</span> : null}
                {selectedRisk.unavailable > 0 ? <span className="sa-risk-error">{selectedRisk.unavailable} unavailable</span> : null}
              </div>
            ) : null}
            <button type="button" className="library-btn primary sa-run-btn" disabled={!canRun} onClick={runBacktest}>
              {running ? "Running..." : "Backtest"}
            </button>
            {!canRun && selectedIds.length > 0 && selectedIds.length > maxComponents ? (
              <div className="sa-callout sa-callout-danger">
                Selection exceeds limit ({selectedIds.length} &gt; {maxComponents}). Reduce selection or increase the limit.
              </div>
            ) : null}
            {!canRun && selectedRisk.reproducible > 0 && !allowReproduceFallback ? (
              <div className="sa-callout sa-callout-warn">
                {selectedRisk.reproducible} component(s) require reproduce fallback. Enable it in Parameters or generate their signal first.
              </div>
            ) : null}
            {selectedIds.length === 1 ? (
              <div className="sa-callout sa-callout-info">Single-factor run is useful for smoke tests; two or more factors are recommended.</div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="field-panel">
        <div className="superalpha-panel-title">
          <strong>Superalpha Runs</strong>
          <span>Select a previous run to reopen detail</span>
        </div>
        <div className="superalpha-run-list">
          {!runs.length ? <span className="metrics-empty">No superalpha runs yet</span> : null}
          {runs.map((run) => {
            const isEditing = editingRunId === run.superalpha_id;
            const displayName = run.display_name || run.name || shortSuperalphaId(run.superalpha_id);
            return (
              <article key={run.superalpha_id} className={selectedRunId === run.superalpha_id ? "superalpha-run-card active" : "superalpha-run-card"}>
                {isEditing ? (
                  <div className="superalpha-rename-row">
                    <input
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      maxLength={80}
                      autoFocus
                      onKeyDown={(event) => {
                        if (event.key === "Enter") saveRename(run);
                        if (event.key === "Escape") cancelRename();
                      }}
                    />
                    <button type="button" title="Save name" disabled={renamingRunId === run.superalpha_id || !renameValue.trim()} onClick={() => saveRename(run)}>
                      <Check size={14} />
                    </button>
                    <button type="button" title="Cancel rename" onClick={cancelRename}>
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <div className="superalpha-run-card-top">
                    <button type="button" className="superalpha-run-select" onClick={() => setSelectedRunId(run.superalpha_id)}>
                      <strong title={displayName}>{displayName}</strong>
                      <span>{run.created_at_utc?.slice(0, 16).replace("T", " ") || run.superalpha_id}</span>
                    </button>
                    <button type="button" className="superalpha-icon-button" title="Rename superalpha" onClick={() => beginRename(run)}>
                      <Pencil size={14} />
                    </button>
                  </div>
                )}
                <button type="button" className="superalpha-run-meta" onClick={() => setSelectedRunId(run.superalpha_id)}>
                  <small>{run.component_count ?? 0} factors</small>
                  <small>score {formatNumber(Number(run.summary?.score_total), 1)}</small>
                  <small>{run.status || "ok"}</small>
                </button>
                <button type="button" className="superalpha-run-combo" title={run.combo_expression || "1"} onClick={() => setSelectedRunId(run.superalpha_id)}>
                  combo: {compactCombo(run.combo_expression)}
                </button>
                <div className="superalpha-run-live-row">
                  <span className={`signal-badge ${liveStatusFor(run.superalpha_id) === "active" ? "badge-ok" : "badge-warn"}`}>
                    {liveStatusFor(run.superalpha_id)}
                  </span>
                  <button
                    type="button"
                    className="compact-button"
                    disabled={activatingRunId === run.superalpha_id || liveStatusFor(run.superalpha_id) === "active"}
                    onClick={() => activateRunForLive(run)}
                  >
                    {activatingRunId === run.superalpha_id ? "Activating..." : "Activate for Live"}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
        {selectedRun ? (
          <div className="superalpha-composition">
            <div className="superalpha-composition-head">
              <div>
                <strong>{selectedRun.display_name || selectedRun.name || shortSuperalphaId(selectedRun.superalpha_id)}</strong>
                <span>{selectedRun.superalpha_id}</span>
              </div>
              <dl>
                <div>
                  <dt>Factors</dt>
                  <dd>{selectedRun.component_count ?? selectedRunComponents.length}</dd>
                </div>
                <div>
                  <dt>Join</dt>
                  <dd>{selectedRun.component_join || "concat"}</dd>
                </div>
                <div>
                  <dt>Score</dt>
                  <dd>{formatNumber(Number(selectedRun.summary?.score_total), 1)}</dd>
                </div>
              </dl>
            </div>
            <div className="superalpha-combo-row">
              <span>Combo</span>
              <code>{selectedRun.combo_expression || "1"}</code>
            </div>
            <div className="superalpha-diagnostics">
              <div>
                <span>Resource</span>
                <strong>{summaryText(selectedRun.resource_summary)}</strong>
              </div>
              <div>
                <span>Cache</span>
                <strong>{summaryText(selectedRun.cache_summary)}</strong>
              </div>
              <div>
                <span>Cleanup</span>
                <strong>{summaryText(selectedRun.cleanup_summary)}</strong>
              </div>
            </div>
            <div className="superalpha-composition-table-wrap">
              <table className="field-table superalpha-composition-table">
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th>Weight</th>
                    <th>Status</th>
                    <th>Mode</th>
                    <th>Score</th>
                    <th>LO Sharpe</th>
                    <th>LS Sharpe</th>
                    <th>Expression</th>
                  </tr>
                </thead>
                <tbody>
                  {!selectedRunComponents.length ? (
                    <tr><td colSpan={8} className="empty-cell">No component metadata</td></tr>
                  ) : null}
                  {selectedRunComponents.map((row, index) => (
                    <tr key={`${row.factor || "component"}-${index}`}>
                      <td className="field-name">{row.factor || "-"}</td>
                      <td>{formatNumber(Number(row.weight), 4)}</td>
                      <td>{row.signal_status || row.status || "-"}</td>
                      <td>{row.acceptance_mode || "-"}</td>
                      <td>{formatNumber(row.score ?? null, 1)}</td>
                      <td>{formatNumber(row.candidate_long_only_sharpe ?? null, 2)}</td>
                      <td>{formatNumber(row.candidate_long_short_sharpe ?? null, 2)}</td>
                      <td title={row.expression || ""}>{row.expression || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </section>

      <section className="field-panel superalpha-result-panel">
        <div className="drawer-action-bar superalpha-tabs">
          <button type="button" className={resultTab === "pnl" ? "drawer-tab active" : "drawer-tab"} onClick={() => setResultTab("pnl")}>
            <LineChart size={17} /> PnL
          </button>
          <button type="button" className={resultTab === "analysis" ? "drawer-tab active" : "drawer-tab"} onClick={() => setResultTab("analysis")}>
            <Images size={17} /> Analysis Data
          </button>
          <button type="button" className={resultTab === "metrics" ? "drawer-tab active" : "drawer-tab"} onClick={() => setResultTab("metrics")}>
            <BarChart3 size={17} /> Metrics
          </button>
        </div>
        {detail?.pnl?.phase_config?.windows?.some((window) => window.key === "test") ? (
          <div className="phase-toggle-bar">
            <label className="phase-toggle">
              <input type="checkbox" checked={showTestPhase} onChange={(event) => setShowTestPhase(event.target.checked)} />
              <span>Show test period</span>
            </label>
          </div>
        ) : null}
        {!selectedRunId ? <div className="chart-empty">Run or select a superalpha to view detail</div> : null}
        {selectedRunId && resultTab === "pnl" ? (
          <PnlChart
            rows={detail?.pnl?.rows ?? []}
            status={detail?.pnl?.status ?? "missing"}
            loading={detailLoading}
            phaseConfig={detail?.pnl?.phase_config ?? null}
            showTestPhase={showTestPhase}
            benchmarkStatus={detail?.pnl?.benchmark_status ?? detail?.run?.benchmark_status ?? null}
          />
        ) : null}
        {selectedRunId && resultTab === "analysis" ? (
          <AnalysisDataPanel
            response={detail?.analysis_data ?? null}
            loading={detailLoading}
            fallbackResponse={null}
            fallbackLoading={false}
          />
        ) : null}
        {selectedRunId && resultTab === "metrics" && detail ? (
          <MetricSummary
            factor={detail.factor}
            run={detail.run}
            phaseMetrics={detail.pnl?.phase_metrics ?? null}
            phaseConfig={detail.pnl?.phase_config ?? null}
            portfolioMetrics={detail.pnl?.portfolio_metrics ?? null}
            showTestPhase={showTestPhase}
          />
        ) : null}
      </section>
    </section>
  );
}
