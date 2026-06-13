import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Database, LineChart, PlayCircle, RefreshCw, ShieldAlert } from "lucide-react";

import { fetchDashboardOverview, fetchPreflight } from "../api";
import type { DashboardOverviewResponse, PreflightResponse } from "../types";
import { errorMessage, isAbortError } from "../utils/format";
import { STATUS_COPY } from "../utils/statusCopy";

type OverviewPageProps = {
  onNavigate: (tab: "alphas" | "data" | "research" | "superalpha" | "live" | "closedLoop") => void;
  onOpenDataHealth?: () => void;
};

export function OverviewPage({ onNavigate, onOpenDataHealth }: OverviewPageProps) {
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(null);
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    Promise.all([fetchDashboardOverview(controller.signal), fetchPreflight(controller.signal)])
      .then(([overviewPayload, preflightPayload]) => {
        setOverview(overviewPayload);
        setPreflight(preflightPayload);
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [refreshKey]);

  const warnings = overview?.freshness_warnings ?? [];
  const preflightWarnings = preflight?.warnings ?? [];

  return (
    <section className="data-page overview-page">
      <header className="data-header">
        <div>
          <strong>Workspace Overview</strong>
          <span>Local status, freshness, and safe next actions</span>
        </div>
        <dl>
          <div>
            <dt>Universes</dt>
            <dd>{overview?.universe_count ?? 0}</dd>
          </div>
          <div>
            <dt>Runs</dt>
            <dd>{overview?.run_count ?? 0}</dd>
          </div>
          <div>
            <dt>Latest Analysis</dt>
            <dd>{formatDate(overview?.latest_analysis_at_utc)}</dd>
          </div>
        </dl>
      </header>

      {error ? <div className="data-error">{STATUS_COPY.error.api}: {error}</div> : null}

      <section className="overview-actions">
        <button type="button" onClick={() => onNavigate("closedLoop")}>
          <PlayCircle size={18} />
          <strong>Run Closed Loop</strong>
          <span>Start mining in an isolated background job.</span>
        </button>
        <button type="button" onClick={() => onNavigate("alphas")}>
          <LineChart size={18} />
          <strong>Review Alphas</strong>
          <span>Browse scoreboard, PnL, and factor diagnostics.</span>
        </button>
        <button type="button" onClick={() => onNavigate("data")}>
          <Database size={18} />
          <strong>Inspect Data</strong>
          <span>Check field catalog coverage and availability.</span>
        </button>
        <button type="button" onClick={() => setRefreshKey((value) => value + 1)}>
          <RefreshCw size={18} />
          <strong>Refresh Status</strong>
          <span>Reload local artifacts and preflight checks.</span>
        </button>
      </section>

      <section className="overview-grid">
        <StatusPanel
          title="Data Freshness"
          status={warnings.length ? "warn" : "ok"}
          rows={[
            ["Field catalog", overview?.field_catalog_status || "missing"],
            ["Max available date", overview?.field_catalog_max_available_end || "-"],
            ["Catalog rows", String(overview?.field_catalog_row_count ?? 0)]
          ]}
          messages={warnings.map((item) => `${item.universe ? `${item.universe}: ` : ""}${item.message}`)}
          actionLabel="Open Data Health"
          onAction={onOpenDataHealth}
        />
        <StatusPanel
          title="System Checks"
          status={preflightWarnings.length ? "warn" : "ok"}
          rows={[
            ["Status", preflight?.status || "-"],
            ["Warnings", String(preflightWarnings.length)],
            ["Strict exit", String(preflight?.strict_exit_code ?? "-")]
          ]}
          messages={[...preflightWarnings, ...(preflight?.remediations ?? [])]}
        />
      </section>
    </section>
  );
}

function StatusPanel({
  title,
  status,
  rows,
  messages,
  actionLabel,
  onAction
}: {
  title: string;
  status: "ok" | "warn";
  rows: Array<[string, string]>;
  messages: string[];
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <section className="field-panel overview-status-panel">
      <div className="superalpha-panel-title">
        <strong>{title}</strong>
        <span className={status === "ok" ? "status-ok" : "status-warn"}>
          {status === "ok" ? <CheckCircle2 size={14} /> : <ShieldAlert size={14} />}
          {status === "ok" ? "OK" : "Review"}
        </span>
      </div>
      <div className="live-state-list">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {messages.length ? (
        <div className="overview-message-list">
          {messages.map((message, index) => (
            <p key={`${message}-${index}`}>
              <AlertTriangle size={14} />
              {message}
            </p>
          ))}
        </div>
      ) : null}
      {onAction ? (
        <button type="button" className="compact-button" onClick={onAction}>{actionLabel || "Open"}</button>
      ) : null}
    </section>
  );
}

function formatDate(value?: string | null): string {
  if (!value) return "-";
  return value.slice(0, 19).replace("T", " ");
}
