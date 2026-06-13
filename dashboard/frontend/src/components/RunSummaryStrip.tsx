import { Images, LineChart, Sigma } from "lucide-react";

import type { AnalysisRun, FactorMetric } from "../types";
import { formatNumber } from "../utils/format";

type RunSummaryStripProps = {
  run: AnalysisRun | null;
  factors: FactorMetric[];
  total: number;
};

export function RunSummaryStrip({ run, factors, total }: RunSummaryStripProps) {
  const tierCounts = factors.reduce<Record<string, number>>((counts, factor) => {
    const tier = factor.effectiveness_tier || "-";
    counts[tier] = (counts[tier] ?? 0) + 1;
    return counts;
  }, {});
  const bestScore = factors.reduce<number | null>((best, factor) => {
    const score = factor.feedback_score ?? factor.score_total;
    if (score === null || score === undefined) {
      return best;
    }
    return best === null ? score : Math.max(best, score);
  }, null);

  return (
    <section className="run-summary-strip">
      <div className="summary-stat">
        <Sigma size={17} />
        <span>Loaded</span>
        <strong>
          {factors.length.toLocaleString()} / {total.toLocaleString()}
        </strong>
      </div>
      <div className="summary-stat">
        <span>Best Feedback Score</span>
        <strong>{formatNumber(bestScore, 1)}</strong>
      </div>
      <div className="summary-tiers" aria-label="Loaded tier distribution">
        {["S", "A", "B", "C"].map((tier) => (
          <span key={tier}>
            {tier}: {tierCounts[tier] ?? 0}
          </span>
        ))}
      </div>
      <div className="summary-badges">
        <span className={run?.has_portfolio_pnl ? "run-badge ready" : "run-badge missing"}>
          <LineChart size={15} />
          {run?.has_portfolio_pnl ? "PnL ready" : "PnL missing"}
        </span>
        <span className={run?.has_analysis_data || run?.has_visualizations ? "run-badge ready" : "run-badge missing"}>
          <Images size={15} />
          {run?.has_analysis_data ? "Analysis ready" : run?.has_visualizations ? "PNG ready" : "Analysis missing"}
        </span>
      </div>
    </section>
  );
}
