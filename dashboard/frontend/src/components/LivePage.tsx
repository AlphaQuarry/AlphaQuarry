import { useEffect, useMemo, useState } from "react";
import { Pause, Play, RefreshCw, RotateCcw } from "lucide-react";

import {
  fetchActiveLiveSuperalphas,
  fetchLiveDataStatus,
  fetchLiveHoldings,
  fetchLiveOrders,
  fetchLiveStatus,
  updateLiveSuperalphaStatus
} from "../api";
import type {
  LiveActiveResponse,
  LiveDataStatusResponse,
  LiveHoldingsResponse,
  LiveOrdersResponse,
  LiveStatusResponse,
  UniverseSummary
} from "../types";
import { errorMessage, formatNumber, isAbortError } from "../utils/format";

function cell(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isFinite(value) ? formatNumber(value, Math.abs(value) < 1 ? 4 : 2) : "-";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

function nested(row: Record<string, unknown> | null | undefined, key: string): unknown {
  if (!row) return undefined;
  const value = row[key];
  return value;
}

export function LivePage({
  universe,
  universes,
  onUniverseChange
}: {
  universe: string;
  universes: UniverseSummary[];
  onUniverseChange: (value: string) => void;
}) {
  const [status, setStatus] = useState<LiveStatusResponse | null>(null);
  const [active, setActive] = useState<LiveActiveResponse | null>(null);
  const [dataStatus, setDataStatus] = useState<LiveDataStatusResponse | null>(null);
  const [holdings, setHoldings] = useState<LiveHoldingsResponse | null>(null);
  const [orders, setOrders] = useState<LiveOrdersResponse | null>(null);
  const [selectedSa, setSelectedSa] = useState("");
  const [detailTab, setDetailTab] = useState<"holdings" | "orders">("holdings");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!universe) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    Promise.all([
      fetchLiveStatus(universe, controller.signal),
      fetchActiveLiveSuperalphas(universe, controller.signal),
      fetchLiveDataStatus(universe, controller.signal)
    ])
      .then(([statusPayload, activePayload, dataPayload]) => {
        setStatus(statusPayload);
        setActive(activePayload);
        setDataStatus(dataPayload);
        setSelectedSa((current) => current || activePayload.superalphas[0]?.superalpha_id || "");
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [universe, refreshKey]);

  useEffect(() => {
    if (!universe || !selectedSa) {
      setHoldings(null);
      setOrders(null);
      return;
    }
    const controller = new AbortController();
    setError("");
    Promise.all([
      fetchLiveHoldings(universe, selectedSa, 200, controller.signal),
      fetchLiveOrders(universe, selectedSa, 500, controller.signal)
    ])
      .then(([holdingsPayload, ordersPayload]) => {
        setHoldings(holdingsPayload);
        setOrders(ordersPayload);
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [universe, selectedSa, refreshKey]);

  const selectedStatus = useMemo(() => {
    const rows = status?.superalphas ?? [];
    return rows.find((row) => row.superalpha_id === selectedSa) ?? null;
  }, [status, selectedSa]);

  const updateStatus = (superalphaId: string, nextStatus: string) => {
    updateLiveSuperalphaStatus(universe, superalphaId, nextStatus)
      .then(() => setRefreshKey((value) => value + 1))
      .catch((exc: unknown) => setError(errorMessage(exc)));
  };

  return (
    <section className="data-page live-page">
      <header className="data-header">
        <div>
          <strong>Live</strong>
          <span>Shadow target holdings and production-readiness state</span>
        </div>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>{status?.status || "missing"}</dd>
          </div>
          <div>
            <dt>Active SA</dt>
            <dd>{active?.total ?? 0}</dd>
          </div>
          <div>
            <dt>Ready Date</dt>
            <dd>{dataStatus?.resolved_signal_date || dataStatus?.common_ready_date || "-"}</dd>
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
          <button type="button" className="compact-button" onClick={() => setRefreshKey((value) => value + 1)}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </section>

      {error ? <div className="data-error">{error}</div> : null}

      <section className="live-grid">
        <div className="field-panel">
          <div className="superalpha-panel-title">
            <strong>Active Superalphas</strong>
            <span>{loading ? "Loading..." : `${active?.total ?? 0} tracked`}</span>
          </div>
          <div className="field-table-wrap live-table-wrap">
            <table className="field-table live-table">
              <thead>
                <tr>
                  <th>SA</th>
                  <th>Status</th>
                  <th>Activated</th>
                  <th>Snapshot</th>
                  <th>Last</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {!active?.superalphas.length ? <tr><td colSpan={6} className="empty-cell">No active live Superalpha</td></tr> : null}
                {active?.superalphas.map((row) => (
                  <tr key={row.superalpha_id} className={selectedSa === row.superalpha_id ? "live-selected-row" : undefined}>
                    <td className="field-name">
                      <button type="button" className="link-button" onClick={() => setSelectedSa(row.superalpha_id)}>
                        {row.display_name || row.superalpha_id}
                      </button>
                    </td>
                    <td>{row.status}</td>
                    <td>{row.activated_at_utc?.slice(0, 16).replace("T", " ") || "-"}</td>
                    <td>{row.source_meta_exists === false ? "source missing" : `${row.snapshot?.component_factor_ids?.length ?? 0} factors`}</td>
                    <td>{row.last_execute_date || row.last_signal_date || "-"}</td>
                    <td>
                      {row.status === "active" ? (
                        <button type="button" className="superalpha-icon-button" title="Pause" onClick={() => updateStatus(row.superalpha_id, "paused")}>
                          <Pause size={14} />
                        </button>
                      ) : (
                        <button type="button" className="superalpha-icon-button" title="Resume" onClick={() => updateStatus(row.superalpha_id, "active")}>
                          <Play size={14} />
                        </button>
                      )}
                      <button type="button" className="superalpha-icon-button" title="Retire" onClick={() => updateStatus(row.superalpha_id, "retired")}>
                        <RotateCcw size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="field-panel">
          <div className="superalpha-panel-title">
            <strong>Run State</strong>
            <span>{selectedSa || "No SA selected"}</span>
          </div>
          <div className="live-state-list">
            <div><span>Global updated</span><strong>{status?.updated_at_utc?.slice(0, 19).replace("T", " ") || "-"}</strong></div>
            <div><span>Selected status</span><strong>{cell(selectedStatus?.status)}</strong></div>
            <div><span>Stale</span><strong>{cell(selectedStatus?.stale)}</strong></div>
            <div><span>Error</span><strong>{cell(selectedStatus?.error)}</strong></div>
            <div><span>Holdings</span><strong>{holdings?.status || "missing"}</strong></div>
            <div><span>Orders</span><strong>{orders?.status || "missing"}</strong></div>
            <div><span>Orders Note</span><strong>{cell((nested(selectedStatus, "orders") as Record<string, unknown> | undefined)?.reason)}</strong></div>
          </div>
        </div>
      </section>

      <section className="field-panel">
        <div className="superalpha-panel-title">
          <strong>Data Status</strong>
          <span>{dataStatus?.status || "missing"}</span>
        </div>
        <div className="live-state-list live-state-inline">
          <div><span>Requested</span><strong>{dataStatus?.requested_date || "-"}</strong></div>
          <div><span>Common ready</span><strong>{dataStatus?.common_ready_date || "-"}</strong></div>
          <div><span>Resolved signal</span><strong>{dataStatus?.resolved_signal_date || "-"}</strong></div>
          <div><span>Blocking</span><strong>{dataStatus?.blocking_fields?.join(", ") || "-"}</strong></div>
          <div><span>Market Value</span><strong>{cell((dataStatus as Record<string, unknown> | null)?.selected_market_value_field)}</strong></div>
          <div><span>MV Coverage</span><strong>{cell((dataStatus as Record<string, unknown> | null)?.selected_market_value_non_null_rate)}</strong></div>
          <div><span>Catalog</span><strong>{cell((dataStatus as Record<string, unknown> | null)?.catalog_status)}</strong></div>
          <div><span>Catalog Warnings</span><strong>{cell(((dataStatus as Record<string, unknown> | null)?.catalog_warnings as string[] | undefined)?.join(", "))}</strong></div>
        </div>
      </section>

      <section className="field-panel">
        <div className="superalpha-panel-title">
          <strong>{detailTab === "holdings" ? "Latest Holdings" : "Orders Review"}</strong>
          <span>{detailTab === "holdings" ? `${holdings?.rows.length ?? 0} rows` : `${orders?.rows.length ?? 0} rows`}</span>
        </div>
        <div className="live-tabbar" role="tablist" aria-label="Live details">
          <button type="button" className={detailTab === "holdings" ? "live-tab-active" : ""} onClick={() => setDetailTab("holdings")}>Holdings</button>
          <button type="button" className={detailTab === "orders" ? "live-tab-active" : ""} onClick={() => setDetailTab("orders")}>Orders</button>
        </div>
        {detailTab === "orders" ? (
          <>
            <div className="live-state-list live-state-inline">
              <div><span>Account Value</span><strong>{cell(orders?.account.account_total_value)}</strong></div>
              <div><span>Cash</span><strong>{cell(orders?.account.cash)}</strong></div>
              <div><span>Position Date</span><strong>{cell(orders?.account.position_date)}</strong></div>
              <div><span>Turnover</span><strong>{cell(orders?.summary.estimated_turnover)}</strong></div>
              <div><span>Fee</span><strong>{cell(orders?.summary.estimated_fee)}</strong></div>
              <div><span>Blocked</span><strong>{cell(orders?.summary.blocked_buy_count)} / {cell(orders?.summary.blocked_sell_count)}</strong></div>
              <div><span>Reviewable</span><strong>{cell(orders?.summary.orders_reviewable)}</strong></div>
              <div><span>Dry Run</span><strong>{cell(orders?.latest?.dry_run)}</strong></div>
              <div><span>Status Note</span><strong>{cell(orders?.latest?.reason || (nested(selectedStatus, "orders") as Record<string, unknown> | undefined)?.reason || orders?.summary.review_block_reason)}</strong></div>
            </div>
            <div className="field-table-wrap live-table-wrap">
              <table className="field-table live-table">
                <thead>
                  <tr>
                    <th>Code</th>
                    <th>Side</th>
                    <th>Shares</th>
                    <th>Value</th>
                    <th>Delta W</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {!orders?.rows.length ? <tr><td colSpan={6} className="empty-cell">No orders artifact available</td></tr> : null}
                  {orders?.rows.map((row, index) => (
                    <tr key={`${row.code}-${index}`}>
                      <td className="field-name">{cell(row.code)}</td>
                      <td>{cell(row.side)}</td>
                      <td>{cell(row.order_shares)}</td>
                      <td>{cell(row.order_value)}</td>
                      <td>{cell(row.delta_weight)}</td>
                      <td>{cell(row.blocked_reason)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="field-table-wrap live-table-wrap">
            <table className="field-table live-table">
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Signal</th>
                  <th>Rank</th>
                  <th>Weight</th>
                  <th>Blocked</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {!holdings?.rows.length ? <tr><td colSpan={6} className="empty-cell">No holdings artifact available</td></tr> : null}
                {holdings?.rows.map((row, index) => (
                  <tr key={`${row.code}-${index}`}>
                    <td className="field-name">{cell(row.code)}</td>
                    <td>{cell(row.signal)}</td>
                    <td>{cell(row.signal_rank)}</td>
                    <td>{cell(row.target_weight)}</td>
                    <td>{cell(row.blocked)}</td>
                    <td>{cell(row.block_reason)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  );
}
