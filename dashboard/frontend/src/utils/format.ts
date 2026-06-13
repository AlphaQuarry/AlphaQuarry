import type { AnalysisRun } from "../types";

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export function formatPermille(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 10000).toFixed(2)} bp`;
}

export function formatRunLabel(run: AnalysisRun): string {
  if (run.is_scoreboard) {
    return `${run.label ?? "All analyzed factors / Scoreboard"} | ${run.factor_count.toLocaleString()} factors`;
  }
  const created = run.created_at_utc ? run.created_at_utc.slice(0, 16).replace("T", " ") : run.run_id.slice(-24);
  const pnl = run.has_portfolio_pnl ? "pnl ok" : "pnl missing";
  const analysis = run.has_analysis_data ? "analysis ok" : run.has_visualizations ? "png ok" : "analysis missing";
  return `${created} | P${run.period} L${run.layers} | ${run.factor_count} factors | ${pnl} | ${analysis}`;
}

export function displayPortfolio(portfolio: string): string {
  if (portfolio === "long_short") return "Long-short";
  if (portfolio === "long_only") return "Long-only";
  if (portfolio === "long_10") return "Long-10";
  if (portfolio === "benchmark") return "Benchmark";
  if (portfolio.startsWith("layer_")) return portfolio.replace("_", " ");
  return portfolio;
}

export function displayCategory(category: string): string {
  const labels: Record<string, string> = {
    distribution: "Distribution",
    ic: "IC",
    layer: "Layer"
  };
  return labels[category] ?? category;
}

export function sortPortfolio(a: string, b: string): number {
  const rank = (value: string) => {
    if (value.startsWith("layer_")) return Number(value.replace("layer_", "")) || 0;
    if (value === "long_short") return 100;
    if (value === "long_only") return 101;
    if (value === "long_10") return 102;
    if (value === "benchmark") return 103;
    return 200;
  };
  return rank(a) - rank(b);
}

export function errorMessage(exc: unknown): string {
  if (exc instanceof Error) {
    return exc.message;
  }
  return String(exc);
}

export function isAbortError(exc: unknown): boolean {
  return exc instanceof DOMException && exc.name === "AbortError";
}
