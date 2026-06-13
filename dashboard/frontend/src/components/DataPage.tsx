import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Search } from "lucide-react";

import { fetchDataFamilies, fetchDataFields, fetchDataHealth } from "../api";
import type { DataFamilySummary, DataField, DataFamiliesResponse, DataFieldsResponse, DataHealthResponse } from "../types";
import { errorMessage, formatPercent, isAbortError } from "../utils/format";
import { STATUS_COPY } from "../utils/statusCopy";

const DATA_PAGE_SIZE = 50;

export function DataPage({ universe, initialView = "catalog" }: { universe: string; initialView?: "catalog" | "health" }) {
  const [familiesPayload, setFamiliesPayload] = useState<DataFamiliesResponse | null>(null);
  const [fieldsPayload, setFieldsPayload] = useState<DataFieldsResponse | null>(null);
  const [healthPayload, setHealthPayload] = useState<DataHealthResponse | null>(null);
  const [view, setView] = useState<"catalog" | "health">(initialView);
  const [selectedFamily, setSelectedFamily] = useState("");
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [searchableOnly, setSearchableOnly] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setView(initialView);
  }, [initialView]);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query), 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    fetchDataFamilies()
      .then((payload) => {
        setFamiliesPayload(payload);
        setSelectedFamily((current) =>
          current && payload.families.some((family) => family.family === current)
            ? current
            : payload.families[0]?.family ?? ""
        );
      })
      .catch((exc: unknown) => setError(errorMessage(exc)));
  }, []);

  const families = familiesPayload?.families ?? [];
  const fields = fieldsPayload?.fields ?? [];
  const total = fieldsPayload?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / DATA_PAGE_SIZE));
  const sourceStatus = fieldsPayload ?? familiesPayload;
  const sourceLabel = useMemo(() => sourceStatus?.source || "field catalog not found", [sourceStatus]);

  useEffect(() => {
    const controller = new AbortController();
    if (!fields.length) setLoading(true);
    setError("");
    fetchDataFields({
      family: selectedFamily,
      q: debouncedQuery,
      searchableOnly,
      limit: DATA_PAGE_SIZE,
      offset: pageIndex * DATA_PAGE_SIZE,
      signal: controller.signal
    })
      .then(setFieldsPayload)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [selectedFamily, debouncedQuery, searchableOnly, pageIndex]);

  useEffect(() => {
    if (view !== "health" || !universe) return;
    const controller = new AbortController();
    setError("");
    fetchDataHealth(universe, controller.signal)
      .then(setHealthPayload)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [view, universe]);

  useEffect(() => {
    setPageIndex(0);
  }, [selectedFamily, debouncedQuery, searchableOnly]);

  return (
    <section className="data-page">
      <header className="data-header">
        <div>
          <strong>Data Catalog</strong>
          <span>{sourceStatus?.status === "ok" ? "Read-only local catalog" : sourceStatus?.message ?? "No catalog available"}</span>
        </div>
        <dl>
          <div>
            <dt>Rows</dt>
            <dd>{(sourceStatus?.row_count ?? 0).toLocaleString()}</dd>
          </div>
          <div>
            <dt>DuckDB</dt>
            <dd>{sourceStatus?.duckdb_path ?? "-"}</dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>{sourceLabel}</dd>
          </div>
        </dl>
      </header>

      {error ? <div className="data-error">{STATUS_COPY.error.api}: {error}</div> : null}

      <section className="view-switch">
        <button type="button" className={view === "catalog" ? "active" : ""} onClick={() => setView("catalog")}>Catalog</button>
        <button type="button" className={view === "health" ? "active" : ""} onClick={() => setView("health")}>Health</button>
      </section>

      {view === "catalog" ? <div className="data-layout">
        <aside className="family-list" aria-label="factor families">
          <button type="button" className={selectedFamily === "" ? "active" : ""} onClick={() => setSelectedFamily("")}>
            <span>All</span>
            <strong>{(familiesPayload?.row_count ?? 0).toLocaleString()}</strong>
          </button>
          {families.map((family) => (
            <FamilyButton
              key={family.family}
              family={family}
              active={selectedFamily === family.family}
              onClick={() => setSelectedFamily(family.family)}
            />
          ))}
        </aside>

        <section className="field-panel">
          <div className="field-toolbar">
            <label className="field-search">
              <Search size={15} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search fields" />
            </label>
            <label className="searchable-toggle">
              <input
                type="checkbox"
                checked={searchableOnly}
                onChange={(event) => setSearchableOnly(event.target.checked)}
              />
              Searchable only
            </label>
          </div>

          <div className="field-table-wrap">
            <table className="field-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Category</th>
                  <th>Source Table</th>
                  <th>Role</th>
                  <th>Dtype</th>
                  <th>Available At</th>
                  <th>Preprocess</th>
                  <th>Date Range</th>
                  <th>Coverage</th>
                  <th>Finite Rate</th>
                  <th>Searchable</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={12} className="empty-cell">{STATUS_COPY.loading.fieldCatalog}</td></tr>
                ) : null}
                {!loading && !fields.length ? (
                  <tr><td colSpan={12} className="empty-cell">No fields</td></tr>
                ) : null}
                {!loading && fields.map((field) => <FieldRow key={field.field_name} field={field} />)}
              </tbody>
            </table>
          </div>

          <footer className="data-pagination">
            <span>{total.toLocaleString()} fields</span>
            <div>
              <button type="button" disabled={loading || pageIndex <= 0} onClick={() => setPageIndex((value) => value - 1)}>
                Prev
              </button>
              <strong>Page {Math.min(pageIndex + 1, totalPages)} / {totalPages}</strong>
              <button type="button" disabled={loading || pageIndex + 1 >= totalPages} onClick={() => setPageIndex((value) => value + 1)}>
                Next
              </button>
            </div>
          </footer>
        </section>
      </div> : <HealthView payload={healthPayload} />}
    </section>
  );
}

function HealthView({ payload }: { payload: DataHealthResponse | null }) {
  if (!payload) {
    return <div className="metrics-empty">{STATUS_COPY.loading.dataHealth}</div>;
  }
  const catalog = payload.catalog;
  const base = payload.universe_base;
  const closedLoop = payload.closed_loop_health;
  const quality = payload.quality_artifact;
  return (
    <section className="data-health-grid">
      {payload.warnings.length ? (
        <section className="overview-message-list standalone">
          {payload.warnings.map((warning) => (
            <p key={warning.code}><AlertTriangle size={14} /> {warning.message}</p>
          ))}
        </section>
      ) : null}
      <HealthPanel
        title="Catalog"
        rows={[
          ["Rows", catalog.row_count],
          ["Searchable", catalog.searchable_count],
          ["Avg coverage", coverageLabel(catalog)],
          ["Low coverage", catalog.low_coverage_count],
          ["Coverage status", coverageStatusLabel(catalog.coverage_status)],
          ["Threshold", formatPercent(asNumber(payload.thresholds?.low_coverage_threshold))],
          ["Max date", catalog.max_available_end]
        ]}
      />
      <HealthPanel
        title="Universe Base"
        rows={[
          ["Exists", base.exists ? "yes" : "no"],
          ["Rows", base.rows],
          ["Columns", base.columns],
          ["Size", formatBytes(asNumber(base.bytes))],
          ["MTime", String(base.mtime_utc || "-").slice(0, 19).replace("T", " ")]
        ]}
      />
      <HealthPanel
        title="Closed Loop Health"
        rows={[
          ["Records", closedLoop.total_records],
          ["Latest", closedLoop.latest_status],
          ["Hard limits", closedLoop.hard_limit_count],
          ["Memory warnings", closedLoop.memory_warning_count],
          ["Scoreboard max", closedLoop.scoreboard_rows_max]
        ]}
      />
      <HealthPanel
        title="Quality Artifact"
        rows={[
          ["Exists", quality.exists ? "yes" : "no"],
          ["Status", quality.overall_status || "-"],
          ["Warn fields", quality.warn_field_count],
          ["Fail fields", quality.fail_field_count],
          ["Generated", String(quality.generated_at_utc || "-").slice(0, 19).replace("T", " ")]
        ]}
      />
      <section className="field-panel data-health-families">
        <div className="superalpha-panel-title">
          <strong>Family Health</strong>
          <span>{payload.families.length} families</span>
        </div>
        <table className="field-table compact">
          <thead>
            <tr>
              <th>Family</th>
              <th>Fields</th>
              <th>Searchable</th>
              <th>Avg Coverage</th>
              <th>Low Coverage</th>
              <th>Max Date</th>
            </tr>
          </thead>
          <tbody>
            {payload.families.map((family) => (
              <tr key={family.family}>
                <td>{family.family}</td>
                <td>{family.field_count}</td>
                <td>{family.searchable_count}</td>
                <td>{coverageLabel({ coverage_status: family.avg_coverage_rate === null || family.avg_coverage_rate === undefined ? "missing" : "available", avg_coverage_rate: family.avg_coverage_rate })}</td>
                <td>{family.low_coverage_count}</td>
                <td>{family.max_available_end || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </section>
  );
}

function HealthPanel({ title, rows }: { title: string; rows: Array<[string, unknown]> }) {
  return (
    <section className="field-panel">
      <div className="superalpha-panel-title"><strong>{title}</strong></div>
      <div className="live-state-list">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{String(value ?? "-")}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function FamilyButton({ family, active, onClick }: { family: DataFamilySummary; active: boolean; onClick: () => void }) {
  return (
    <button type="button" className={active ? "active" : ""} onClick={onClick}>
      <span>{family.family}</span>
      <strong>{family.field_count.toLocaleString()}</strong>
      <small>{formatPercent(family.avg_coverage_rate)}</small>
    </button>
  );
}

function FieldRow({ field }: { field: DataField }) {
  const dateRange = [field.available_start, field.available_end].filter(Boolean).join(" to ") || "-";
  return (
    <tr>
      <td className="field-name">{field.field_name}</td>
      <td>{field.category || "-"}</td>
      <td>{field.source_table || "-"}</td>
      <td>{field.field_role || "-"}</td>
      <td>{field.dtype || field.field_type || "-"}</td>
      <td>{field.available_at || "-"}</td>
      <td>{field.preprocessing_policy || "-"}</td>
      <td>{dateRange}</td>
      <td>{formatPercent(field.coverage_rate ?? null)}</td>
      <td>{formatPercent(field.finite_rate ?? null)}</td>
      <td>{field.is_searchable ? "Yes" : "No"}</td>
      <td>{field.description || "-"}</td>
    </tr>
  );
}

function asNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function coverageLabel(catalog: Record<string, unknown>): string {
  if (catalog.coverage_status === "missing") return STATUS_COPY.missing.notRefreshed;
  return formatPercent(asNumber(catalog.avg_coverage_rate));
}

function coverageStatusLabel(value: unknown): string {
  if (value === "available") return "Available";
  if (value === "partial") return "Partial";
  if (value === "missing") return STATUS_COPY.missing.notRefreshed;
  return "-";
}

function formatBytes(value: number | null): string {
  if (value === null) return "-";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
