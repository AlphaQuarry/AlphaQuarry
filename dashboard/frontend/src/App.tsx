import { useEffect, useMemo, useState } from "react";
import type { SortingState } from "@tanstack/react-table";
import { Activity, Database, Gauge, LineChart, PlayCircle, Radio, Sparkles, X } from "lucide-react";

import { fetchFactorAnalysisData, fetchFactorVisualizations, fetchFactors, fetchPnl, fetchRuns, fetchUniverses } from "./api";
import { ClosedLoopPage } from "./components/ClosedLoopPage";
import { DataPage } from "./components/DataPage";
import { FactorDrawer, type DrawerTab } from "./components/FactorDrawer";
import { FactorTable } from "./components/FactorTable";
import { OverviewPage } from "./components/OverviewPage";
import { RunSummaryStrip } from "./components/RunSummaryStrip";
import { RunToolbar } from "./components/RunToolbar";
import { ResearchPage } from "./components/ResearchPage";
import { LivePage } from "./components/LivePage";
import { SuperalphaPage } from "./components/SuperalphaPage";
import type {
  AnalysisRun,
  AnalysisDataResponse,
  FactorMetric,
  PnlResponse,
  SortDir,
  UniverseSummary,
  VisualizationResponse
} from "./types";
import { errorMessage, isAbortError } from "./utils/format";

const DEFAULT_PAGE_SIZE = 20;
type MainTab = "overview" | "alphas" | "data" | "research" | "superalpha" | "live" | "closedLoop";

export default function App() {
  const [universes, setUniverses] = useState<UniverseSummary[]>([]);
  const [selectedUniverse, setSelectedUniverse] = useState("");
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [factors, setFactors] = useState<FactorMetric[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [effectiveOnly, setEffectiveOnly] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([{ id: "feedback_score", desc: true }]);
  const [selectedFactor, setSelectedFactor] = useState<FactorMetric | null>(null);
  const [pnl, setPnl] = useState<PnlResponse | null>(null);
  const [pnlLoading, setPnlLoading] = useState(false);
  const [showTestPhase, setShowTestPhase] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("pnl");
  const [visuals, setVisuals] = useState<VisualizationResponse | null>(null);
  const [visualsLoading, setVisualsLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisDataResponse | null>(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [activeMainTab, setActiveMainTab] = useState<MainTab>("overview");
  const [dataInitialView, setDataInitialView] = useState<"catalog" | "health">("catalog");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId]
  );
  const detailRunId = useMemo(() => {
    if (!selectedRun?.is_scoreboard) {
      return selectedRunId;
    }
    return String(selectedFactor?.analysis_run_id ?? "");
  }, [selectedRun, selectedRunId, selectedFactor]);
  const sortBy = sorting[0]?.id ?? "feedback_score";
  const sortDir: SortDir = sorting[0]?.desc ? "desc" : "asc";

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query), 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    setError("");
    fetchUniverses(refreshKey > 0)
      .then((rows) => {
        setUniverses(rows);
        if (!selectedUniverse || !rows.some((row) => row.name === selectedUniverse)) {
          setSelectedUniverse(rows[0]?.name ?? "");
        }
      })
      .catch((exc: unknown) => setError(errorMessage(exc)));
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedUniverse) {
      setRuns([]);
      setSelectedRunId("");
      return;
    }
    setRuns([]);
    setSelectedRunId("");
    setError("");
    fetchRuns(selectedUniverse, refreshKey > 0)
      .then((rows) => {
        setRuns(rows);
        setSelectedRunId((current) => (rows.some((run) => run.run_id === current) ? current : rows[0]?.run_id ?? ""));
      })
      .catch((exc: unknown) => setError(errorMessage(exc)));
  }, [selectedUniverse, refreshKey]);

  useEffect(() => {
    if (!selectedUniverse || !selectedRunId || !selectedRun) {
      setFactors([]);
      setTotal(0);
      return;
    }
    const controller = new AbortController();
    if (!factors.length) setLoading(true);
    setError("");
    fetchFactors({
      universe: selectedUniverse,
      runId: selectedRunId,
      q: debouncedQuery,
      sortBy,
      sortDir,
      effectiveOnly,
      limit: pageSize,
      offset: pageIndex * pageSize,
      signal: controller.signal
    })
      .then((payload) => {
        setFactors(payload.factors);
        setTotal(payload.total);
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [selectedUniverse, selectedRunId, selectedRun, debouncedQuery, sortBy, sortDir, effectiveOnly, pageIndex, pageSize, refreshKey]);

  useEffect(() => {
    setSelectedFactor(null);
    setPnl(null);
    setVisuals(null);
    setAnalysisData(null);
    setDrawerTab("pnl");
    setShowTestPhase(false);
    setFactors([]);
    setTotal(0);
    setPageIndex(0);
  }, [selectedUniverse, selectedRunId]);

  useEffect(() => {
    setPageIndex(0);
  }, [debouncedQuery, sortBy, sortDir, effectiveOnly, refreshKey, pageSize]);

  useEffect(() => {
    if (!selectedFactor || !selectedUniverse || !selectedRunId || !selectedRun) {
      setPnl(null);
      setPnlLoading(false);
      return;
    }
    if (selectedRun.is_scoreboard && !detailRunId) {
      setPnl({
        status: "missing",
        factor: selectedFactor.factor,
        rows: [],
        message: "This factor does not have an analysis_run_id in the scoreboard; detail artifacts cannot be resolved."
      });
      setPnlLoading(false);
      return;
    }
    const controller = new AbortController();
    setPnl(null);
    setPnlLoading(true);
    setError("");
    fetchPnl(selectedUniverse, detailRunId, selectedFactor.factor, controller.signal, showTestPhase)
      .then(setPnl)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => setPnlLoading(false));
    return () => controller.abort();
  }, [selectedFactor, selectedUniverse, selectedRunId, selectedRun, detailRunId, showTestPhase]);

  useEffect(() => {
    if (drawerTab !== "images" || !selectedFactor || !selectedUniverse || !selectedRunId || !selectedRun) {
      return;
    }
    if (selectedRun.is_scoreboard && !detailRunId) {
      const message = "This factor does not have an analysis_run_id in the scoreboard; detail artifacts cannot be resolved.";
      setAnalysisData({
        status: "missing",
        factor: selectedFactor.factor,
        message
      } as AnalysisDataResponse);
      setVisuals({
        status: "missing",
        factor: selectedFactor.factor,
        images: [],
        message
      });
      return;
    }
    const controller = new AbortController();
    setAnalysisDataLoading(true);
    setVisualsLoading(true);
    setAnalysisData(null);
    setVisuals(null);
    setError("");
    Promise.all([
      fetchFactorAnalysisData(selectedUniverse, detailRunId, selectedFactor.factor, controller.signal, showTestPhase),
      fetchFactorVisualizations(selectedUniverse, detailRunId, selectedFactor.factor, controller.signal)
    ])
      .then(([analysisPayload, visualsPayload]) => {
        setAnalysisData(analysisPayload);
        setVisuals(visualsPayload);
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) {
          setError(errorMessage(exc));
        }
      })
      .finally(() => {
        setAnalysisDataLoading(false);
        setVisualsLoading(false);
      });
    return () => controller.abort();
  }, [drawerTab, selectedFactor, selectedUniverse, selectedRunId, selectedRun, detailRunId, showTestPhase]);

  const runStatus = selectedRun
    ? `${selectedRun.has_dashboard_metrics ? "Dashboard metrics" : selectedRun.has_factor_metrics ? "Factor metrics" : "Metrics missing"} - ${
        selectedRun.created_at_utc || selectedRun.run_id
      }`
    : null;

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="brand-mark">A</div>
        <div className="brand-copy">
          <strong>Factor Dashboard</strong>
          <span>Local alpha universe review</span>
        </div>
        <div className="nav-tabs">
          <button type="button" className={activeMainTab === "overview" ? "active-tab" : ""} onClick={() => setActiveMainTab("overview")}>
            <Gauge size={16} /> Overview
          </button>
          <button type="button" className={activeMainTab === "closedLoop" ? "active-tab" : ""} onClick={() => setActiveMainTab("closedLoop")}>
            <PlayCircle size={16} /> Closed Loop
          </button>
          <button type="button" className={activeMainTab === "alphas" ? "active-tab" : ""} onClick={() => setActiveMainTab("alphas")}>
            <LineChart size={16} /> Alphas
          </button>
          <button type="button" className={activeMainTab === "data" ? "active-tab" : ""} onClick={() => {
            setDataInitialView("catalog");
            setActiveMainTab("data");
          }}>
            <Database size={16} /> Data
          </button>
          <button type="button" className={activeMainTab === "research" ? "active-tab" : ""} onClick={() => setActiveMainTab("research")}>
            <Activity size={16} /> Research
          </button>
          <button type="button" className={activeMainTab === "superalpha" ? "active-tab" : ""} onClick={() => setActiveMainTab("superalpha")}>
            <Sparkles size={16} /> Superalpha
          </button>
          <button type="button" className={activeMainTab === "live" ? "active-tab" : ""} onClick={() => setActiveMainTab("live")}>
            <Radio size={16} /> Live
          </button>
        </div>
      </header>

      <main className="workspace">
        {error ? (
          <div className="error-strip">
            <span>{error}</span>
            <button type="button" onClick={() => setError("")}><X size={16} /></button>
          </div>
        ) : null}

        {activeMainTab === "overview" ? (
          <OverviewPage
            onNavigate={setActiveMainTab}
            onOpenDataHealth={() => {
              setDataInitialView("health");
              setActiveMainTab("data");
            }}
          />
        ) : null}
        {activeMainTab === "closedLoop" ? (
          <ClosedLoopPage universe={selectedUniverse} universes={universes} onUniverseChange={setSelectedUniverse} />
        ) : null}
        {activeMainTab === "alphas" ? (
          <>
            <RunToolbar
              universes={universes}
              selectedUniverse={selectedUniverse}
              runs={runs}
              selectedRunId={selectedRunId}
              query={query}
              effectiveOnly={effectiveOnly}
              onUniverseChange={setSelectedUniverse}
              onRunChange={setSelectedRunId}
              onQueryChange={setQuery}
              onEffectiveOnlyChange={() => setEffectiveOnly((value) => !value)}
              onRefresh={() => setRefreshKey((value) => value + 1)}
            />
            <RunSummaryStrip run={selectedRun} factors={factors} total={total} />
            <FactorTable
              factors={factors}
              total={total}
              loading={loading}
              sorting={sorting}
              selectedFactor={selectedFactor}
              runStatus={runStatus}
              effectiveOnly={effectiveOnly}
              pageIndex={pageIndex}
              pageSize={pageSize}
              onSortingChange={setSorting}
              onSelectFactor={(factor) => {
                setShowTestPhase(false);
                setAnalysisData(null);
                setVisuals(null);
                setSelectedFactor(factor);
              }}
              onPageChange={setPageIndex}
              onPageSizeChange={setPageSize}
            />
          </>
        ) : null}
        {activeMainTab === "data" ? <DataPage universe={selectedUniverse} initialView={dataInitialView} /> : null}
        {activeMainTab === "research" ? (
          <ResearchPage universe={selectedUniverse} universes={universes} onUniverseChange={setSelectedUniverse} />
        ) : null}
        {activeMainTab === "superalpha" ? (
          <SuperalphaPage universe={selectedUniverse} universes={universes} onUniverseChange={setSelectedUniverse} />
        ) : null}
        {activeMainTab === "live" ? (
          <LivePage universe={selectedUniverse} universes={universes} onUniverseChange={setSelectedUniverse} />
        ) : null}
      </main>

      {activeMainTab === "alphas" && selectedFactor ? (
        <FactorDrawer
          factor={selectedFactor}
          universe={selectedUniverse}
          detailRunId={detailRunId}
          pnl={pnl}
          pnlLoading={pnlLoading}
          run={selectedRun}
          activeTab={drawerTab}
          visuals={visuals}
          visualsLoading={visualsLoading}
          analysisData={analysisData}
          analysisDataLoading={analysisDataLoading}
          showTestPhase={showTestPhase}
          onShowTestPhaseChange={setShowTestPhase}
          onTabChange={setDrawerTab}
          onLibraryChanged={() => setRefreshKey((value) => value + 1)}
          onClose={() => setSelectedFactor(null)}
        />
      ) : null}
    </div>
  );
}
