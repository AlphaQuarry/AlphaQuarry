import { BarChart3 } from "lucide-react";

import type {
  AnalysisRun,
  FactorMetric,
  PhaseConfig,
  PhaseMetrics,
  PhaseWindow,
  PortfolioMetricsResponse
} from "../types";
import { formatNumber, formatPercent, formatPermille } from "../utils/format";

type MetricGroup = {
  title: string;
  metrics: Array<{ label: string; value: string }>;
};

type PhaseRow = {
  key: string;
  label: string;
  scope: string;
  values: Record<string, unknown>;
};

const CORE_METRICS = [
  { key: "score_total", label: "Score", format: "number1" },
  { key: "ic_mean", label: "IC", format: "number4" },
  { key: "ir", label: "IR", format: "number2" },
  { key: "long_short_total_return", label: "L/S Return", format: "percent" },
  { key: "long_short_sharpe_ratio", label: "L/S Sharpe", format: "number2" },
  { key: "long_short_max_drawdown", label: "L/S Drawdown", format: "percent" },
  { key: "turnover_long_short_mean", label: "L/S Turnover", format: "percent" },
  { key: "margin_long_short", label: "L/S Margin", format: "permille" }
] as const;

export function MetricSummary({
  factor,
  run,
  phaseMetrics,
  phaseConfig,
  portfolioMetrics,
  showTestPhase
}: {
  factor: FactorMetric;
  run: AnalysisRun | null;
  phaseMetrics?: PhaseMetrics | null;
  phaseConfig?: PhaseConfig | null;
  portfolioMetrics?: PortfolioMetricsResponse | null;
  showTestPhase: boolean;
}) {
  const phaseRows = buildPhaseRows(factor, phaseMetrics, phaseConfig, showTestPhase);
  const groups = buildFullPeriodGroups(factor);

  return (
    <section className="metric-summary">
      <div className="expression-block compact">
        <span>Expression</span>
        <code>{factor.expression || "-"}</code>
      </div>

      <div className="summary-title compact">
        <BarChart3 size={22} />
        <span>Phase Metrics</span>
        <small>P{run?.period ?? factor.period} - L{run?.layers ?? factor.layers}</small>
      </div>
      <div className="metrics-scope-note">
        Primary cards use visible train / validation phases. Test appears only when Show test period is enabled.
      </div>

      <div className="phase-card-grid">
        {phaseRows.map((row) => (
          <PhaseMetricCard key={row.key} row={row} />
        ))}
      </div>

      <PortfolioBreakdownTable portfolioMetrics={portfolioMetrics ?? null} />

      <section className="full-period-section polished">
        <div className="phase-metric-title">
          <span>Full Period Compatibility</span>
          <strong>All available dates, may include test</strong>
        </div>
        <div className="phase-metric-table-wrap full-period-table-wrap">
          <table className="phase-metric-table full-period-table">
            <colgroup>
              <col className="full-period-group-col" />
              <col />
              <col className="full-period-value-col" />
            </colgroup>
            <thead>
              <tr>
                <th>Group</th>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {groups.flatMap((group) =>
                group.metrics.map((metric, index) => (
                  <tr
                    key={`${group.title}-${metric.label}`}
                    className={index === 0 ? "full-period-group-start" : undefined}
                  >
                    {index === 0 ? (
                      <td className="full-period-group-cell" rowSpan={group.metrics.length}>
                        {group.title}
                      </td>
                    ) : null}
                    <td>{metric.label}</td>
                    <td>{metric.value}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}

function PhaseMetricCard({ row }: { row: PhaseRow }) {
  return (
    <article className={`phase-metric-card phase-${row.key}`}>
      <header>
        <span>{row.label}</span>
        <small>{row.scope}</small>
      </header>
      <div className="phase-kpi-strip">
        {CORE_METRICS.map((metric) => (
          <Metric
            key={`${row.key}-${metric.key}`}
            label={metric.label}
            value={formatMetricValue(
              phaseMetricValue(row, metric.key),
              metric.format
            )}
          />
        ))}
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PortfolioBreakdownTable({ portfolioMetrics }: { portfolioMetrics?: PortfolioMetricsResponse | null }) {
  const rows = portfolioMetrics?.rows ?? [];
  return (
    <section className="phase-metric-section portfolio-breakdown-section">
      <div className="phase-metric-title">
        <span>Phase Breakdown</span>
        <strong>Feedback Scope: {phaseLabel(portfolioMetrics?.scope_phase || "train")}</strong>
      </div>
      {!rows.length ? <div className="metrics-empty">No portfolio breakdown is available for the feedback phase.</div> : null}
      <div className="phase-metric-table-wrap">
        <table className="phase-metric-table">
          <thead>
            <tr>
              <th>Portfolio</th>
              <th>Total Return</th>
              <th>Annual Return</th>
              <th>Excess Annual</th>
              <th>Annual Volatility</th>
              <th>Max Drawdown</th>
              <th>Turnover</th>
              <th>Sharpe</th>
              <th>Fitness</th>
              <th>Obs</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.portfolio}>
                <td>{row.label || row.portfolio}</td>
                <td>{formatPercent(row.total_return)}</td>
                <td>{formatPercent(row.annualized_return)}</td>
                <td>{formatPercent(row.excess_annualized_return)}</td>
                <td>{formatPercent(row.annualized_volatility)}</td>
                <td>{formatPercent(row.max_drawdown)}</td>
                <td>{formatPercent(row.turnover)}</td>
                <td>{formatNumber(row.sharpe, 2)}</td>
                <td>{formatNumber(row.fitness, 2)}</td>
                <td>{formatNumber(row.obs ?? null, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function buildPhaseRows(
  factor: FactorMetric,
  phaseMetrics?: PhaseMetrics | null,
  phaseConfig?: PhaseConfig | null,
  showTestPhase = false
): PhaseRow[] {
  const windows = visiblePhaseWindows(phaseMetrics, phaseConfig, showTestPhase);
  return windows.map((window) => ({
    key: String(window.key),
    label: phaseLabel(String(window.key)),
    scope: phaseScope(window),
    values: phaseValues(String(window.key), factor, phaseMetrics)
  }));
}

function visiblePhaseWindows(
  phaseMetrics?: PhaseMetrics | null,
  phaseConfig?: PhaseConfig | null,
  showTestPhase = false
): PhaseWindow[] {
  const configured = phaseConfig?.windows ?? [];
  const source =
    configured.length > 0
      ? configured
      : (["train", "val", "test"] as const)
          .filter((key) => Boolean(phaseMetrics?.[key]))
          .map((key) => ({ key, label: phaseLabel(key), start: "" }));
  return source.filter((window) => {
    const key = String(window.key);
    if (key === "test" && !showTestPhase) {
      return false;
    }
    return ["train", "val", "test"].includes(key);
  });
}

function phaseValues(phase: string, factor: FactorMetric, phaseMetrics?: PhaseMetrics | null): Record<string, unknown> {
  const fromResponse = phaseMetrics?.[phase];
  if (fromResponse && typeof fromResponse === "object") {
    return fromResponse as Record<string, unknown>;
  }
  const out: Record<string, unknown> = {};
  for (const suffix of [
    "obs",
    "ic_mean",
    "ir",
    "positive_ic_ratio",
    "long_short_total_return",
    "long_short_annualized_return",
    "long_short_volatility",
    "long_short_sharpe_ratio",
    "long_short_max_drawdown",
    "long_short_fitness_ratio",
    "turnover_long_short_mean",
    "margin_long_short",
    "margin_long_short_bp",
    "turnover_long_only_mean",
    "margin_long_only",
    "score_total",
    "feedback_score"
  ]) {
    out[suffix] = factor[`${phase}_${suffix}`];
  }
  return out;
}

function phaseMetricValue(row: PhaseRow, key: (typeof CORE_METRICS)[number]["key"]): number | null {
  return asNumber(row.values[key]);
}

function buildFullPeriodGroups(factor: FactorMetric): MetricGroup[] {
  const benchmarkExcess = asNumber(factor.best_minus_benchmark_annualized_return);
  const universeExcess = asNumber(factor.best_minus_universe_annualized_return);
  return [
    {
      title: "Predictive",
      metrics: [
        { label: "IC", value: formatNumber(factor.ic_mean, 4) },
        { label: "IR", value: formatNumber(factor.ir, 2) }
      ]
    },
    {
      title: "Long-short",
      metrics: [
        { label: "Total Return", value: formatPercent(factor.long_short_total_return) },
        { label: "Annual Return", value: formatPercent(factor.long_short_annualized_return) },
        { label: "Sharpe", value: formatNumber(factor.long_short_sharpe_ratio, 2) },
        { label: "Drawdown", value: formatPercent(factor.long_short_max_drawdown) },
        { label: "Fitness", value: formatNumber(factor.long_short_fitness_ratio, 2) }
      ]
    },
    {
      title: "Layer",
      metrics: [
        { label: "Best Total", value: formatPercent(factor.best_layer_total_return) },
        { label: "Best Annual", value: formatPercent(factor.best_layer_annualized_return) },
        { label: "Best Sharpe", value: formatNumber(factor.best_layer_sharpe, 2) },
        { label: "Best Drawdown", value: formatPercent(factor.best_layer_max_drawdown) },
        { label: "Best Fitness", value: formatNumber(factor.best_layer_fitness_ratio, 2) },
        {
          label: benchmarkExcess !== null ? "Vs Benchmark" : "Vs Universe",
          value: formatPercent(benchmarkExcess !== null ? benchmarkExcess : universeExcess)
        }
      ]
    },
    {
      title: "Tradeability",
      metrics: [
        { label: "LO Turnover", value: formatPercent(factor.turnover_long_only_mean) },
        { label: "LO Margin", value: formatPermille(factor.margin_long_only) }
      ]
    },
    {
      title: "Scores",
      metrics: [
        { label: "Predictive", value: formatNumber(factor.score_predictive_power, 1) },
        { label: "Performance", value: formatNumber(factor.score_long_only_performance, 1) },
        { label: "Stability", value: formatNumber(factor.score_stability, 1) },
        { label: "Tradeability", value: formatNumber(factor.score_tradeability, 1) },
        { label: "Total", value: formatNumber(factor.score_total, 1) },
        { label: "Tier", value: factor.effectiveness_tier ?? "-" }
      ]
    }
  ];
}

function phaseLabel(phase: string): string {
  if (phase === "train") return "Feedback / Train";
  if (phase === "val") return "Validation";
  if (phase === "test") return "Test";
  return phase;
}

function phaseScope(window: PhaseWindow): string {
  const start = String(window.start || "");
  const end = String(window.end || "");
  if (start && end) {
    return `${start} to ${end}`;
  }
  if (start) {
    return `from ${start}`;
  }
  return "phase metrics";
}

function formatMetricValue(value: number | null, format: (typeof CORE_METRICS)[number]["format"]): string {
  if (format === "percent") {
    return formatPercent(value);
  }
  if (format === "permille") {
    return formatPermille(value);
  }
  if (format === "number4") {
    return formatNumber(value, 4);
  }
  if (format === "number2") {
    return formatNumber(value, 2);
  }
  return formatNumber(value, 1);
}

function asNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
