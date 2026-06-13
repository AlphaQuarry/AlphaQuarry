import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import * as echarts from "echarts";

import type { PhaseConfig, PhaseWindow, PnlRow } from "../types";
import { displayPortfolio, formatPercent, sortPortfolio } from "../utils/format";

type PnlMode = "gross" | "net";

export function PnlChart({
  rows,
  status,
  loading,
  phaseConfig,
  showTestPhase,
  benchmarkStatus
}: {
  rows: PnlRow[];
  status: string;
  loading: boolean;
  phaseConfig?: PhaseConfig | null;
  showTestPhase: boolean;
  benchmarkStatus?: Record<string, unknown> | null;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const resizeCleanupRef = useRef<(() => void) | null>(null);
  const visiblePhaseWindows = useMemo(
    () => (phaseConfig?.windows ?? []).filter((window) => showTestPhase || window.key !== "test"),
    [phaseConfig, showTestPhase]
  );
  const chartRows = useMemo(
    () => filterRowsForVisiblePhases(rows, visiblePhaseWindows),
    [rows, visiblePhaseWindows]
  );
  const portfolios = useMemo(() => Array.from(new Set(chartRows.map((row) => row.portfolio))).sort(sortPortfolio), [chartRows]);
  const [pnlMode, setPnlMode] = useState<PnlMode>("gross");
  const [visible, setVisible] = useState<Set<string>>(new Set());
  const portfolioNetAvailability = useMemo(() => {
    const availability = new Map<string, boolean>();
    for (const row of chartRows) {
      if (Boolean(row.has_net_pnl) && row.cum_return_net !== null && row.cum_return_net !== undefined) {
        availability.set(row.portfolio, true);
      } else if (!availability.has(row.portfolio)) {
        availability.set(row.portfolio, false);
      }
    }
    return availability;
  }, [chartRows]);
  const netAvailable = useMemo(
    () => Array.from(portfolioNetAvailability.values()).some(Boolean),
    [portfolioNetAvailability]
  );
  const netCostSummary = useMemo(
    () => buildNetCostSummary(chartRows, visible, portfolioNetAvailability),
    [chartRows, visible, portfolioNetAvailability]
  );
  const benchmarkCode = useMemo(() => benchmarkCodeFromStatus(benchmarkStatus), [benchmarkStatus]);

  useEffect(() => {
    const primary = portfolios.filter((portfolio) => !portfolio.startsWith("layer_"));
    setVisible(new Set(primary.length ? primary : portfolios));
  }, [portfolios.join("|")]);

  useEffect(() => {
    if (pnlMode !== "net") {
      return;
    }
    setVisible((current) => new Set(Array.from(current).filter((portfolio) => portfolioNetAvailability.get(portfolio))));
  }, [pnlMode, portfolioNetAvailability]);

  useEffect(() => {
    return () => {
      resizeCleanupRef.current?.();
      resizeCleanupRef.current = null;
      chartInstanceRef.current?.dispose();
      chartInstanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current || chartRows.length === 0) {
      resizeCleanupRef.current?.();
      resizeCleanupRef.current = null;
      chartInstanceRef.current?.dispose();
      chartInstanceRef.current = null;
      return;
    }
    const chart = ensureChart(chartRef.current, chartInstanceRef, resizeCleanupRef);
    const maxDate = chartRows.reduce<string | undefined>(
      (current, row) => (!current || row.trade_date > current ? row.trade_date : current),
      undefined
    );
    const phaseMarkArea = buildPhaseMarkArea(visiblePhaseWindows, maxDate);
    const phaseMarkLine = buildPhaseMarkLine(visiblePhaseWindows);
    const phaseSeries =
      phaseMarkArea.length || phaseMarkLine.length
        ? [
            {
              name: "__phase_windows__",
              id: "__phase_windows__",
              type: "line",
              data: [],
              silent: true,
              symbol: "none",
              tooltip: { show: false },
              lineStyle: { opacity: 0 },
              z: -10,
              emphasis: { disabled: true },
              blur: { lineStyle: { opacity: 0 } },
              ...(phaseMarkArea.length ? { markArea: { silent: true, data: phaseMarkArea } } : {}),
              ...(phaseMarkLine.length ? { markLine: { silent: true, symbol: "none", data: phaseMarkLine } } : {})
            }
          ]
        : [];
    const portfolioSeries = portfolios
      .filter((portfolio) => visible.has(portfolio))
      .filter((portfolio) => pnlMode === "gross" || portfolioNetAvailability.get(portfolio))
      .map((portfolio) => {
        const data = chartRows
          .filter((row) => row.portfolio === portfolio)
          .sort((a, b) => a.trade_date.localeCompare(b.trade_date))
          .map((row) => [row.trade_date, rowCumReturn(row, pnlMode)])
          .filter((point): point is [string, number] => point[1] !== null);
        return {
          name: displayPortfolioForChart(portfolio, benchmarkCode),
          id: `portfolio-${portfolio}`,
          type: "line",
          showSymbol: false,
          smooth: false,
          emphasis: { lineStyle: { opacity: 1 } },
          itemStyle: { color: portfolioColor(portfolio) },
          lineStyle: portfolioLineStyle(portfolio),
          data
        };
      });
    const series = [...phaseSeries, ...portfolioSeries];
    chart.setOption({
      animation: false,
      grid: { top: 28, right: 22, bottom: 74, left: 58 },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value: unknown) => formatPercent(Number(value))
      },
      xAxis: { type: "time", max: maxDate, axisLine: { lineStyle: { color: "#c7d7e5" } } },
      yAxis: {
        type: "value",
        axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: "#e5edf5" } }
      },
      dataZoom: [
        {
          type: "slider",
          bottom: 20,
          height: 18,
          borderColor: "transparent",
          backgroundColor: "rgba(148, 163, 184, 0.12)",
          fillerColor: "rgba(45, 156, 202, 0.14)",
          handleSize: 12,
          showDetail: false,
          brushSelect: false
        },
        { type: "inside" }
      ],
      series
    }, { replaceMerge: ["series"] });
  }, [chartRows, portfolios, visible, visiblePhaseWindows, pnlMode, portfolioNetAvailability, benchmarkCode]);

  if (loading) {
    return <div className="chart-empty">Loading PnL...</div>;
  }

  if (status !== "ok" || chartRows.length === 0) {
    return <div className="chart-empty">No PnL data for this run</div>;
  }

  return (
    <section className="chart-section">
      <div className="pnl-mode-bar">
        <span>PnL Mode</span>
        <button
          type="button"
          className={pnlMode === "gross" ? "segmented active" : "segmented"}
          onClick={() => setPnlMode("gross")}
        >
          Gross
        </button>
        <button
          type="button"
          className={pnlMode === "net" ? "segmented active" : "segmented"}
          disabled={!netAvailable}
          onClick={() => setPnlMode("net")}
        >
          After-fee
        </button>
      </div>
      {!netAvailable ? (
        <div className="chart-note">After-fee disabled: this run was generated without transaction-cost PnL.</div>
      ) : pnlMode === "net" ? (
        <div className="chart-note">After-fee mode hides portfolios without net PnL, such as benchmark or diagnostic long_short.</div>
      ) : null}
      {pnlMode === "net" && netCostSummary ? (
        <div className="pnl-cost-summary" aria-label="After-fee PnL cost diagnostics">
          <span>
            <strong>Cost Model</strong>
            {netCostSummary.costModel}
          </span>
          <span>
            <strong>Cumulative Cost</strong>
            {formatPercent(netCostSummary.totalCost)}
          </span>
          <span>
            <strong>Avg Buy Turnover</strong>
            {formatPercent(netCostSummary.avgBuyTurnover)}
          </span>
          <span>
            <strong>Avg Sell Turnover</strong>
            {formatPercent(netCostSummary.avgSellTurnover)}
          </span>
        </div>
      ) : null}
      <div ref={chartRef} className="pnl-chart" />
      {benchmarkStatus?.status === "missing" ? (
        <div className="chart-note">Benchmark unavailable: {String(benchmarkStatus.reason || "index data not found")}</div>
      ) : null}
      <div className="portfolio-switcher">
        {portfolios.map((portfolio) => {
          const disabledInNet = pnlMode === "net" && !portfolioNetAvailability.get(portfolio);
          return (
          <label key={portfolio} className={disabledInNet ? "disabled-option" : ""}>
            <input
              type="checkbox"
              checked={visible.has(portfolio)}
              disabled={disabledInNet}
              onChange={(event) => {
                setVisible((current) => {
                  const next = new Set(current);
                  if (event.target.checked) {
                    next.add(portfolio);
                  } else {
                    next.delete(portfolio);
                  }
                  return next;
                });
              }}
            />
            {displayPortfolio(portfolio)}
          </label>
        )})}
      </div>
    </section>
  );
}

function rowCumReturn(row: PnlRow, mode: PnlMode): number | null {
  if (mode === "net") {
    if (!row.has_net_pnl) return null;
    return row.cum_return_net ?? null;
  }
  return row.cum_return_gross ?? row.cum_return ?? null;
}

function benchmarkCodeFromStatus(status?: Record<string, unknown> | null): string {
  const code = status?.code ?? status?.benchmark_code;
  return typeof code === "string" && code.trim() ? code.trim() : "";
}

function displayPortfolioForChart(portfolio: string, benchmarkCode: string): string {
  if (portfolio === "benchmark" && benchmarkCode) {
    return `Benchmark (${benchmarkCode})`;
  }
  return displayPortfolio(portfolio);
}

function buildNetCostSummary(
  rows: PnlRow[],
  visible: Set<string>,
  netAvailability: Map<string, boolean>
): { costModel: string; totalCost: number; avgBuyTurnover: number | null; avgSellTurnover: number | null } | null {
  const eligible = rows.filter(
    (row) =>
      visible.has(row.portfolio) &&
      Boolean(netAvailability.get(row.portfolio)) &&
      Boolean(row.has_net_pnl)
  );
  if (!eligible.length) {
    return null;
  }
  const costModels = Array.from(new Set(eligible.map((row) => String(row.cost_model || "")).filter(Boolean)));
  const costs = eligible.map((row) => numericOrNull(row.transaction_cost)).filter((value): value is number => value !== null);
  const buys = eligible.map((row) => numericOrNull(row.buy_turnover)).filter((value): value is number => value !== null);
  const sells = eligible.map((row) => numericOrNull(row.sell_turnover)).filter((value): value is number => value !== null);
  return {
    costModel: costModels.length ? costModels.join(", ") : "-",
    totalCost: costs.reduce((sum, value) => sum + value, 0),
    avgBuyTurnover: averageOrNull(buys),
    avgSellTurnover: averageOrNull(sells)
  };
}

function numericOrNull(value: number | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function averageOrNull(values: number[]): number | null {
  if (!values.length) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function ensureChart(
  element: HTMLDivElement,
  chartRef: MutableRefObject<echarts.ECharts | null>,
  resizeCleanupRef: MutableRefObject<(() => void) | null>
) {
  if (chartRef.current) {
    return chartRef.current;
  }
  const chart = echarts.init(element);
  const resize = () => chart.resize();
  window.addEventListener("resize", resize);
  resizeCleanupRef.current = () => window.removeEventListener("resize", resize);
  chartRef.current = chart;
  return chart;
}

function filterRowsForVisiblePhases(rows: PnlRow[], windows: PhaseWindow[]): PnlRow[] {
  if (!windows.length) {
    return rows;
  }
  const visiblePhaseKeys = new Set(windows.map((window) => String(window.key)));
  return rows.filter((row) => {
    const phase = String(row.phase || "");
    if (phase) {
      return visiblePhaseKeys.has(phase);
    }
    return windows.some((window) => dateInWindow(row.trade_date, window));
  });
}

function dateInWindow(date: string, window: PhaseWindow): boolean {
  if (!date || !window.start) {
    return false;
  }
  if (date < String(window.start)) {
    return false;
  }
  if (window.end && date > String(window.end)) {
    return false;
  }
  return true;
}

function buildPhaseMarkArea(windows: PhaseWindow[], maxDate?: string) {
  return windows
    .map((window) => {
      const end = window.end || maxDate;
      if (!window.start || !end) {
        return null;
      }
      return [
        {
          name: window.label,
          xAxis: window.start,
          itemStyle: { color: phaseColor(window.key), opacity: 0.18 },
          label: {
            show: true,
            position: "insideTop",
            color: phaseLabelColor(window.key),
            fontWeight: 700,
            formatter: window.label
          }
        },
        { xAxis: end }
      ];
    })
    .filter(Boolean);
}

function buildPhaseMarkLine(windows: PhaseWindow[]) {
  return windows
    .filter((window) => window.key !== "train")
    .map((window) => ({
      name: window.label,
      xAxis: window.start,
      lineStyle: { color: phaseLabelColor(window.key), type: "dashed", width: 1.2, opacity: 0.78 },
      label: { formatter: window.label, color: phaseLabelColor(window.key) }
    }));
}

function phaseColor(key: string): string {
  if (key === "val") return "#f8e7a1";
  if (key === "test") return "#f4b7bd";
  return "#cde7e4";
}

function phaseLabelColor(key: string): string {
  if (key === "val") return "#9a7112";
  if (key === "test") return "#9f3342";
  return "#23726c";
}

function portfolioLineStyle(portfolio: string) {
  if (portfolio === "benchmark") {
    return { color: portfolioColor(portfolio), width: 1.35, type: "dashed", opacity: 0.72 };
  }
  if (portfolio.startsWith("layer_")) {
    return { color: portfolioColor(portfolio), width: 0.9, opacity: 0.88 };
  }
  return { color: portfolioColor(portfolio), width: 1.65, opacity: 0.96 };
}

function portfolioColor(portfolio: string): string {
  if (portfolio === "long_short") return "#2563eb";
  if (portfolio === "long_only") return "#46b94a";
  if (portfolio === "long_10") return "#27c6d6";
  if (portfolio === "benchmark") return "#64748b";
  if (portfolio.startsWith("layer_")) {
    const index = Number(portfolio.replace("layer_", ""));
    const colors = [
      "#0f766e",
      "#15803d",
      "#65a30d",
      "#ca8a04",
      "#f59e0b",
      "#ea580c",
      "#dc2626",
      "#c026d3",
      "#7c3aed",
      "#1d4ed8"
    ];
    return colors[Math.max(0, Math.min(colors.length - 1, index - 1))];
  }
  return "#94a3b8";
}
