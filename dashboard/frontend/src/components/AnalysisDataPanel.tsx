import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";

import type {
  AnalysisDataResponse,
  AnalysisCoveragePoint,
  AnalysisDistributionBin,
  AnalysisIcDecayPoint,
  AnalysisLayerTerminalReturn,
  VisualizationResponse
} from "../types";
import { formatNumber, formatPercent } from "../utils/format";
import { AnalysisImageGallery } from "./AnalysisImageGallery";

type AnalysisDataPanelProps = {
  response: AnalysisDataResponse | null;
  loading: boolean;
  fallbackResponse: VisualizationResponse | null;
  fallbackLoading: boolean;
};

const PHASE_ORDER = ["train", "val", "test", "full"];

export function AnalysisDataPanel({ response, loading, fallbackResponse, fallbackLoading }: AnalysisDataPanelProps) {
  if (loading) {
    return <div className="image-empty">Loading analysis data...</div>;
  }
  if (!response || response.status !== "ok") {
    return <AnalysisImageGallery response={fallbackResponse} loading={fallbackLoading} />;
  }

  const hasCharts = Boolean(
    response.ic_series.length ||
      response.distribution.length ||
      response.ic_distribution.length ||
      response.ic_decay.length ||
      response.yearly_ic.length ||
      (response.monthly_ic ?? []).length ||
      (response.coverage_series ?? []).length ||
      response.layer_terminal_return.length
  );
  if (!hasCharts) {
    return <AnalysisImageGallery response={fallbackResponse} loading={fallbackLoading} />;
  }

  return (
    <section className="analysis-data">
      <div className="analysis-data-header">
        <strong>Analysis Data</strong>
        <span>Backend-computed, phase-aware</span>
      </div>
      <div className="analysis-data-grid">
        {response.ic_series.length ? (
          <ChartCard title="Daily IC" option={buildDailyIcOption(response)} wide />
        ) : null}
        {response.ic_series.length ? (
          <ChartCard title="Cumulative IC" option={buildCumulativeIcOption(response)} wide />
        ) : null}
        {(response.coverage_series ?? []).length ? (
          <ChartCard title="Factor Coverage Rate" option={buildCoverageOption(response)} wide />
        ) : null}
        {response.distribution.length ? (
          <ChartCard title="Factor Distribution" option={buildHistogramOption(response.distribution, "Factor bin")} />
        ) : null}
        {response.ic_distribution.length ? (
          <ChartCard title="IC Distribution" option={buildHistogramOption(response.ic_distribution, "IC bin")} />
        ) : null}
        {response.ic_decay.length ? (
          <ChartCard title={`IC Decay${spearmanSuffix(response.ic_decay.map((row) => row.ic_decay_rank_corr))}`} option={buildDecayOption(response)} />
        ) : null}
        {response.layer_terminal_return.length ? (
          <ChartCard
            title={`Layer Terminal Return${spearmanSuffix(response.layer_terminal_return.map((row) => row.rank_corr))}`}
            option={buildLayerTerminalOption(response)}
          />
        ) : null}
        {response.yearly_ic.length ? (
          <ChartCard title="Yearly IC" option={buildYearlyOption(response)} />
        ) : null}
        {(response.monthly_ic ?? []).length ? (
          <ChartCard title="Monthly IC" option={buildMonthlyOption(response)} />
        ) : null}
      </div>
    </section>
  );
}

function ChartCard({ title, option, wide = false }: { title: string; option: echarts.EChartsOption; wide?: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const stableOption = useMemo(() => option, [option]);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    const chart = echarts.init(ref.current);
    chart.setOption(stableOption, true);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [stableOption]);

  return (
    <article className={wide ? "analysis-data-card wide" : "analysis-data-card"}>
      <header>{title}</header>
      <div ref={ref} className="analysis-data-chart" />
    </article>
  );
}

function buildDailyIcOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const overlay = phaseOverlay(response);
  return {
    color: ["#2563eb"],
    grid: { top: 28, right: 18, bottom: 68, left: 48 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatNumber(Number(value), 4) },
    xAxis: { type: "time" },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    dataZoom: subtleDataZoom(),
    series: [
      {
        name: "Daily IC",
        type: "line",
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 0.65 },
        data: response.ic_series.map((row) => [row.trade_date, row.ic ?? null]),
        ...overlay
      }
    ]
  };
}

function buildCumulativeIcOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const overlay = phaseOverlay(response);
  const phases = uniquePhases(response.ic_series);
  return {
    color: ["#16a34a"],
    grid: { top: 28, right: 18, bottom: 68, left: 48 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatNumber(Number(value), 4) },
    xAxis: { type: "time" },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    dataZoom: subtleDataZoom(),
    series: phases.map((phase, index) => ({
        name: `Cumulative IC - ${phaseLabel(phase)}`,
        type: "line",
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 1.15 },
        data: response.ic_series
          .filter((row) => String(row.phase || "full") === phase)
          .map((row) => [row.trade_date, row.cumulative_ic ?? null]),
        ...(index === 0 ? overlay : {})
      }))
  };
}

function buildCoverageOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const rows = trimLeadingZeroCoverageRows(response.coverage_series ?? []);
  const overlay = phaseOverlayFromDates(response, rows.map((row) => row.trade_date));
  return {
    color: ["#23726c"],
    grid: { top: 28, right: 18, bottom: 68, left: 54 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatPercent(Number(value)) },
    xAxis: { type: "time" },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: "#e5edf5" } }
    },
    dataZoom: subtleDataZoom(),
    series: [
      {
        name: "Coverage Rate",
        type: "line",
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 1.1 },
        connectNulls: false,
        data: rows.map((row: AnalysisCoveragePoint) => [row.trade_date, row.coverage_rate ?? null]),
        ...overlay
      }
    ]
  };
}

function trimLeadingZeroCoverageRows(rows: AnalysisCoveragePoint[]): AnalysisCoveragePoint[] {
  const firstPositiveIndex = rows.findIndex((row) => {
    const value = Number(row.coverage_rate);
    return Number.isFinite(value) && value > 0;
  });
  if (firstPositiveIndex <= 0) {
    return rows;
  }
  return rows.slice(firstPositiveIndex);
}

function subtleDataZoom(): echarts.EChartsOption["dataZoom"] {
  return [
    {
      type: "slider",
      bottom: 18,
      height: 16,
      borderColor: "transparent",
      backgroundColor: "rgba(148, 163, 184, 0.10)",
      fillerColor: "rgba(45, 156, 202, 0.12)",
      handleSize: 10,
      showDetail: false,
      brushSelect: false
    },
    { type: "inside" }
  ];
}

function buildHistogramOption(rows: AnalysisDistributionBin[], xName: string): echarts.EChartsOption {
  const phases = uniquePhases(rows);
  const bins = Array.from(new Set(rows.map((row) => String(row.bin_index)))).sort((a, b) => Number(a) - Number(b));
  return {
    color: phaseColors(phases),
    grid: { top: 28, right: 18, bottom: 42, left: 54 },
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => histogramTooltip(params, rows)
    },
    legend: { top: 0, right: 0 },
    xAxis: { type: "category", data: bins, name: "bin" },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    series: phases.map((phase) => ({
      name: phaseLabel(phase),
      type: "bar",
      data: bins.map((bin) => rows.find((row) => row.phase === phase && String(row.bin_index) === bin)?.count ?? 0)
    }))
  };
}

function buildDecayOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const phases = uniquePhases(response.ic_decay);
  const lags = Array.from(new Set(response.ic_decay.map((row) => Number(row.lag)))).sort((a, b) => a - b);
  return {
    color: phaseColors(phases),
    grid: { top: 28, right: 18, bottom: 38, left: 54 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatNumber(Number(value), 4) },
    legend: { top: 0, right: 0 },
    xAxis: { type: "category", data: lags.map(String), name: "lag" },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    series: phases.map((phase) => ({
      name: phaseLabel(phase),
      type: "bar",
      data: lags.map((lag) => response.ic_decay.find((row) => row.phase === phase && Number(row.lag) === lag)?.ic ?? null)
    }))
  };
}

function buildYearlyOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const phases = uniquePhases(response.yearly_ic);
  const years = Array.from(new Set(response.yearly_ic.map((row) => row.year))).sort();
  return {
    color: phaseColors(phases),
    grid: { top: 28, right: 18, bottom: 38, left: 54 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatNumber(Number(value), 4) },
    legend: { top: 0, right: 0 },
    xAxis: { type: "category", data: years },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    series: phases.map((phase) => ({
      name: phaseLabel(phase),
      type: "bar",
      data: years.map((year) => response.yearly_ic.find((row) => row.phase === phase && row.year === year)?.ic_mean ?? null)
    }))
  };
}

function buildMonthlyOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const rows = response.monthly_ic ?? [];
  const phases = uniquePhases(rows);
  const months = Array.from(new Set(rows.map((row) => row.month))).sort();
  return {
    color: phaseColors(phases),
    grid: { top: 28, right: 18, bottom: 46, left: 54 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatNumber(Number(value), 4) },
    legend: { top: 0, right: 0 },
    xAxis: { type: "category", data: months },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5edf5" } } },
    series: phases.map((phase) => ({
      name: phaseLabel(phase),
      type: "bar",
      data: months.map((month) => rows.find((row) => row.phase === phase && row.month === month)?.ic_mean ?? null)
    }))
  };
}

function buildLayerTerminalOption(response: AnalysisDataResponse): echarts.EChartsOption {
  const phases = uniquePhases(response.layer_terminal_return);
  const layers = Array.from(new Set(response.layer_terminal_return.map((row) => row.layer))).sort((a, b) => Number(a) - Number(b));
  return {
    color: phaseColors(phases),
    grid: { top: 28, right: 18, bottom: 40, left: 58 },
    tooltip: { trigger: "axis", valueFormatter: (value: unknown) => formatPercent(Number(value)) },
    legend: { top: 0, right: 0 },
    xAxis: { type: "category", data: layers.map((layer) => `L${layer}`) },
    yAxis: {
      type: "value",
      axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: "#e5edf5" } }
    },
    series: phases.map((phase) => ({
      name: phaseLabel(phase),
      type: "bar",
      data: layers.map((layer) => response.layer_terminal_return.find((row) => row.phase === phase && row.layer === layer)?.terminal_return ?? null)
    }))
  };
}

function phaseOverlay(response: AnalysisDataResponse): Record<string, unknown> {
  const dates = response.ic_series
    .map((row) => Date.parse(row.trade_date))
    .filter((value) => Number.isFinite(value));
  return phaseOverlayFromDates(response, dates.map((value) => isoDate(new Date(value))));
}

function phaseOverlayFromDates(response: AnalysisDataResponse, dateTexts: string[]): Record<string, unknown> {
  const dates = dateTexts
    .map((date) => Date.parse(date))
    .filter((value) => Number.isFinite(value));
  if (!dates.length) {
    return {};
  }
  const minDate = new Date(Math.min(...dates));
  const maxDate = new Date(Math.max(...dates));
  const minText = isoDate(minDate);
  const maxText = isoDate(maxDate);
  const dataPhases = new Set(response.ic_series.map((row) => String(row.phase || "")));
  const windows = (response.phase_config?.windows ?? []).filter((window) => dataPhases.has(String(window.key)));
  const areas = windows
    .map((window) => {
      const start = maxDateText(String(window.start), minText);
      const end = minDateText(String(window.end || maxText), maxText);
      if (!start || !end || Date.parse(start) > Date.parse(end)) {
        return null;
      }
      return [
        {
          name: phaseLabel(String(window.key)),
          xAxis: start,
          itemStyle: { color: phaseFillColor(String(window.key)) },
          label: { show: true, color: phaseTextColor(String(window.key)), fontWeight: 700 }
        },
        { xAxis: end }
      ];
    })
    .filter(Boolean);
  const lines = windows
    .map((window) => String(window.start || ""))
    .filter((start) => start && Date.parse(start) > minDate.getTime() && Date.parse(start) <= maxDate.getTime())
    .map((start) => ({ xAxis: start }));

  return {
    markArea: {
      silent: true,
      data: areas
    },
    markLine: {
      silent: true,
      symbol: "none",
      lineStyle: { color: "#64748b", type: "dashed", width: 1 },
      label: { show: false },
      data: lines
    }
  };
}

function histogramTooltip(params: unknown, rows: AnalysisDistributionBin[]): string {
  const points = Array.isArray(params) ? params : [];
  const binIndex = String((points[0] as { axisValue?: string } | undefined)?.axisValue ?? "");
  const sample = rows.find((row) => String(row.bin_index) === binIndex);
  const range = sample ? `${formatNumber(sample.bin_left, 4)} to ${formatNumber(sample.bin_right, 4)}` : `bin ${binIndex}`;
  const lines = points.map((point) => {
    const item = point as { marker?: string; seriesName?: string; value?: number };
    return `${item.marker ?? ""}${item.seriesName ?? ""}: ${formatNumber(Number(item.value ?? 0), 0)}`;
  });
  return [`${range}`, ...lines].join("<br/>");
}

function uniquePhases<T extends { phase?: string | null }>(rows: T[]): string[] {
  return Array.from(new Set(rows.map((row) => String(row.phase || "full")))).sort(
    (a, b) => phaseOrder(a) - phaseOrder(b)
  );
}

function phaseOrder(phase: string): number {
  const index = PHASE_ORDER.indexOf(phase);
  return index >= 0 ? index : PHASE_ORDER.length;
}

function phaseLabel(phase: string): string {
  if (phase === "train") return "Train";
  if (phase === "val") return "Val";
  if (phase === "test") return "Test";
  return phase;
}

function phaseColors(phases: string[]): string[] {
  const colorMap: Record<string, string> = {
    train: "#23726c",
    val: "#d39b16",
    test: "#d14f61",
    full: "#2563eb"
  };
  return phases.map((phase) => colorMap[phase] ?? "#64748b");
}

function phaseFillColor(phase: string): string {
  const colorMap: Record<string, string> = {
    train: "rgba(35, 114, 108, 0.08)",
    val: "rgba(211, 155, 22, 0.10)",
    test: "rgba(209, 79, 97, 0.10)",
    full: "rgba(37, 99, 235, 0.06)"
  };
  return colorMap[phase] ?? "rgba(100, 116, 139, 0.08)";
}

function phaseTextColor(phase: string): string {
  const colorMap: Record<string, string> = {
    train: "#23726c",
    val: "#9a6a05",
    test: "#b12f43",
    full: "#2563eb"
  };
  return colorMap[phase] ?? "#64748b";
}

function spearmanSuffix(values: Array<number | null | undefined>): string {
  const nums = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  const unique = Array.from(new Set(nums.map((value) => value.toFixed(6)))).map(Number);
  if (!unique.length) {
    return "";
  }
  if (unique.length === 1) {
    return ` (Spearman=${formatNumber(unique[0], 2)})`;
  }
  const avg = unique.reduce((sum, value) => sum + value, 0) / unique.length;
  return ` (Spearman=${formatNumber(avg, 2)} avg)`;
}

function isoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function maxDateText(a: string, b: string): string {
  return Date.parse(a) >= Date.parse(b) ? a : b;
}

function minDateText(a: string, b: string): string {
  return Date.parse(a) <= Date.parse(b) ? a : b;
}
