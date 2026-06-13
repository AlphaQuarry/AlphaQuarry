import { useEffect, useState } from "react";
import { BarChart3, Images, LineChart, X } from "lucide-react";

import { checkFactorLibrary, submitFactorLibrary } from "../api";
import type { AnalysisDataResponse, AnalysisRun, FactorMetric, LibraryCheckResponse, PnlResponse, VisualizationResponse } from "../types";
import { errorMessage, formatNumber } from "../utils/format";
import { AnalysisDataPanel } from "./AnalysisDataPanel";
import { MetricSummary } from "./MetricSummary";
import { PnlChart } from "./PnlChart";

export type DrawerTab = "pnl" | "images" | "metrics";

type FactorDrawerProps = {
  factor: FactorMetric;
  universe: string;
  detailRunId: string;
  pnl: PnlResponse | null;
  pnlLoading: boolean;
  run: AnalysisRun | null;
  activeTab: DrawerTab;
  visuals: VisualizationResponse | null;
  visualsLoading: boolean;
  analysisData: AnalysisDataResponse | null;
  analysisDataLoading: boolean;
  showTestPhase: boolean;
  onShowTestPhaseChange: (value: boolean) => void;
  onTabChange: (tab: DrawerTab) => void;
  onLibraryChanged?: () => void;
  onClose: () => void;
};

export function FactorDrawer({
  factor,
  universe,
  detailRunId,
  pnl,
  pnlLoading,
  run,
  activeTab,
  visuals,
  visualsLoading,
  analysisData,
  analysisDataLoading,
  showTestPhase,
  onShowTestPhaseChange,
  onTabChange,
  onLibraryChanged,
  onClose
}: FactorDrawerProps) {
  const hasTestPhase = Boolean(pnl?.phase_config?.windows?.some((window) => window.key === "test"));
  const [libraryCheck, setLibraryCheck] = useState<LibraryCheckResponse | null>(null);
  const [checking, setChecking] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [libraryMessage, setLibraryMessage] = useState("");

  useEffect(() => {
    setLibraryCheck(null);
    setChecking(false);
    setSubmitting(false);
    setLibraryMessage("");
  }, [factor.factor]);

  const maxCorr = libraryCheck?.corr
    ? Math.max(
        ...["signal_corr", "ic_corr", "max_pnl_corr"]
          .map((key) => Number(libraryCheck.corr?.[key]))
          .filter((value) => Number.isFinite(value))
      )
    : null;
  const canSubmit = Boolean(libraryCheck?.can_submit) && !checking && !submitting;
  const handleCheck = () => {
    if (!universe || !detailRunId || checking) {
      return;
    }
    setChecking(true);
    setLibraryMessage("");
    checkFactorLibrary(universe, detailRunId, factor.factor)
      .then((payload) => {
        setLibraryCheck(payload);
        setLibraryMessage(payload.reason || "");
      })
      .catch((exc: unknown) => setLibraryMessage(errorMessage(exc)))
      .finally(() => setChecking(false));
  };
  const handleSubmit = () => {
    if (!canSubmit || !universe || !detailRunId) {
      return;
    }
    setSubmitting(true);
    setLibraryMessage("");
    submitFactorLibrary(universe, detailRunId, factor.factor)
      .then((payload) => {
        setLibraryMessage(payload.submitted ? `Submitted: ${payload.acceptance_mode || "standard"}` : payload.reason || "Submit blocked");
        if (payload.check) {
          setLibraryCheck(payload.check);
        }
        if (payload.submitted) {
          onLibraryChanged?.();
        }
      })
      .catch((exc: unknown) => setLibraryMessage(errorMessage(exc)))
      .finally(() => setSubmitting(false));
  };

  return (
    <aside className="detail-drawer" aria-label="Factor detail">
      <div className="drawer-header">
        <div>
          <span className="state-pill">LOCAL</span>
          <strong>{factor.factor}</strong>
          <p>{factor.expression}</p>
        </div>
        <button type="button" className="icon-button" title="Close" onClick={onClose}>
          <X size={19} />
        </button>
      </div>
      <div className="drawer-action-bar">
        <button
          type="button"
          className={activeTab === "pnl" ? "drawer-tab active" : "drawer-tab"}
          onClick={() => onTabChange("pnl")}
        >
          <LineChart size={17} />
          PnL
        </button>
        <button
          type="button"
          className={activeTab === "images" ? "drawer-tab active" : "drawer-tab"}
          onClick={() => onTabChange("images")}
        >
          <Images size={17} />
          Analysis Data
        </button>
        <button
          type="button"
          className={activeTab === "metrics" ? "drawer-tab active" : "drawer-tab"}
          onClick={() => onTabChange("metrics")}
        >
          <BarChart3 size={17} />
          Metrics
        </button>
      </div>
      {hasTestPhase ? (
        <div className="phase-toggle-bar">
          <label className="phase-toggle">
            <input
              type="checkbox"
              checked={showTestPhase}
              onChange={(event) => onShowTestPhaseChange(event.target.checked)}
            />
            <span>Show test period</span>
          </label>
          {!showTestPhase ? <span className="phase-toggle-note">Test available, hidden by default</span> : null}
        </div>
      ) : null}
      {activeTab === "pnl" ? (
        <PnlChart
          rows={pnl?.rows ?? []}
          status={pnl?.status ?? "missing"}
          loading={pnlLoading}
          phaseConfig={pnl?.phase_config ?? null}
          showTestPhase={showTestPhase}
          benchmarkStatus={pnl?.benchmark_status ?? run?.benchmark_status ?? null}
        />
      ) : null}
      {activeTab === "images" ? (
        <AnalysisDataPanel
          response={analysisData}
          loading={analysisDataLoading}
          fallbackResponse={visuals}
          fallbackLoading={visualsLoading}
        />
      ) : null}
      {activeTab === "metrics" ? (
        <MetricSummary
          factor={factor}
          run={run}
          phaseMetrics={pnl?.phase_metrics ?? null}
          phaseConfig={pnl?.phase_config ?? null}
          portfolioMetrics={pnl?.portfolio_metrics ?? null}
          showTestPhase={showTestPhase}
        />
      ) : null}
      <div className="drawer-library-footer">
        <div className="library-check-summary">
          {libraryCheck ? (
            <>
              <strong className={`library-decision ${libraryCheck.decision}`}>{libraryCheck.decision.replaceAll("_", " ").toUpperCase()}</strong>
              <span>score {formatNumber(libraryCheck.score, 1)} {libraryCheck.score_basis || ""}</span>
              <span>max corr {formatNumber(Number.isFinite(maxCorr) ? maxCorr : null, 3)}</span>
              <span>nearest {String(libraryCheck.corr?.nearest_factor_id || "-")}</span>
              {libraryCheck.thresholds ? (
                <span>
                  rule score {formatNumber(libraryCheck.thresholds.min_score, 1)} / staging {formatNumber(libraryCheck.thresholds.staging_min_score, 1)}
                </span>
              ) : null}
              {libraryCheck.acceptance_mode === "sharpe_override" ? <span>Sharpe override</span> : null}
            </>
          ) : (
            <span>Check this factor before submitting it to the library.</span>
          )}
          {libraryMessage ? <em>{libraryMessage}</em> : null}
          {libraryCheck?.high_corr_peers?.length ? (
            <details>
              <summary>Details</summary>
              <span>
                {libraryCheck.high_corr_peers
                  .slice(0, 5)
                  .map((row) => `${String(row.peer_factor || "")}: ${formatNumber(Number(row.max_any_corr), 3)}`)
                  .join(" | ")}
              </span>
            </details>
          ) : null}
        </div>
        <div className="library-actions">
          <button type="button" className="library-btn secondary" onClick={handleCheck} disabled={checking || !detailRunId}>
            {checking ? "Checking" : "Check"}
          </button>
          <button type="button" className="library-btn primary" onClick={handleSubmit} disabled={!canSubmit}>
            {submitting ? "Submitting" : "Submit"}
          </button>
        </div>
      </div>
    </aside>
  );
}
