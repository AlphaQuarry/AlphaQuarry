import type {
  AnalysisDataResponse,
  AnalysisRun,
  ClosedLoopJob,
  ClosedLoopJobsResponse,
  ClosedLoopParamsResponse,
  DashboardOverviewResponse,
  DataFamiliesResponse,
  DataFieldsResponse,
  DataHealthResponse,
  FactorMetric,
  LibraryCheckResponse,
  LibraryResponse,
  LibraryStatusResponse,
  LibrarySubmitResponse,
  LiveActivateResponse,
  LiveActiveResponse,
  LiveDataStatusResponse,
  LiveHoldingsResponse,
  LiveOrdersResponse,
  LiveStatusResponse,
  PnlResponse,
  PreflightResponse,
  RunCompareResponse,
  SortDir,
  SuperalphaBacktestRequest,
  SuperalphaBacktestResponse,
  SuperalphaComponentsResponse,
  SuperalphaDetailResponse,
  SuperalphaRenameResponse,
  SuperalphaRunsResponse,
  UniverseSummary,
  VisualizationResponse
} from "./types";

function cleanApiError(raw: string): string {
  if (!raw) return "";
  try {
    const parsed = JSON.parse(raw);
    const detail = parsed?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      return String(detail.message ?? JSON.stringify(detail));
    }
    if (Array.isArray(detail)) {
      return detail.map((item: unknown) => (typeof item === "string" ? item : JSON.stringify(item))).join("; ");
    }
  } catch {
    // not JSON — fall through
  }
  return raw.replace(/\b\w+Error:\s*/g, "").replace(/\s*\(\S+\.py:\d+\)/g, "").trim();
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(cleanApiError(text) || response.statusText);
  }
  return (await response.json()) as T;
}

export async function fetchUniverses(refresh = false): Promise<UniverseSummary[]> {
  const params = new URLSearchParams();
  if (refresh) {
    params.set("refresh", "true");
  }
  const suffix = params.toString() ? `?${params}` : "";
  const payload = await getJson<{ universes: UniverseSummary[] }>(`/api/universes${suffix}`);
  return payload.universes;
}

export async function fetchDashboardOverview(signal?: AbortSignal): Promise<DashboardOverviewResponse> {
  return getJson<DashboardOverviewResponse>("/api/dashboard/overview", { signal });
}

export async function fetchPreflight(signal?: AbortSignal): Promise<PreflightResponse> {
  return getJson<PreflightResponse>("/api/preflight", { signal });
}

export async function fetchRuns(universe: string, refresh = false): Promise<AnalysisRun[]> {
  const params = new URLSearchParams({ universe });
  if (refresh) {
    params.set("refresh", "true");
  }
  const payload = await getJson<{ runs: AnalysisRun[] }>(`/api/runs?${params}`);
  return payload.runs;
}

export async function fetchFactors(options: {
  universe: string;
  runId: string;
  q: string;
  sortBy: string;
  sortDir: SortDir;
  effectiveOnly: boolean;
  limit: number;
  offset: number;
  signal?: AbortSignal;
}): Promise<{ total: number; factors: FactorMetric[]; status: string }> {
  const params = new URLSearchParams({
    universe: options.universe,
    run_id: options.runId,
    q: options.q,
    sort_by: options.sortBy,
    sort_dir: options.sortDir,
    effective_only: String(options.effectiveOnly),
    limit: String(options.limit),
    offset: String(options.offset)
  });
  return getJson<{ total: number; factors: FactorMetric[]; status: string }>(`/api/factors?${params}`, { signal: options.signal });
}

export async function fetchPnl(
  universe: string,
  runId: string,
  factor: string,
  signal?: AbortSignal,
  includeTest = false
): Promise<PnlResponse> {
  const params = new URLSearchParams({ universe, run_id: runId, include_test: String(includeTest) });
  return getJson<PnlResponse>(`/api/factors/${encodeURIComponent(factor)}/pnl?${params}`, { signal });
}

export async function fetchFactorVisualizations(
  universe: string,
  runId: string,
  factor: string,
  signal?: AbortSignal
): Promise<VisualizationResponse> {
  const params = new URLSearchParams({ universe, run_id: runId });
  return getJson<VisualizationResponse>(
    `/api/factors/${encodeURIComponent(factor)}/visualizations?${params}`,
    { signal }
  );
}

export async function fetchFactorAnalysisData(
  universe: string,
  runId: string,
  factor: string,
  signal?: AbortSignal,
  includeTest = false
): Promise<AnalysisDataResponse> {
  const params = new URLSearchParams({ universe, run_id: runId, include_test: String(includeTest) });
  return getJson<AnalysisDataResponse>(
    `/api/factors/${encodeURIComponent(factor)}/analysis-data?${params}`,
    { signal }
  );
}

export async function fetchDataFamilies(): Promise<DataFamiliesResponse> {
  return getJson<DataFamiliesResponse>("/api/data/families");
}

export async function fetchDataFields(options: {
  family: string;
  q: string;
  searchableOnly: boolean;
  limit: number;
  offset: number;
  signal?: AbortSignal;
}): Promise<DataFieldsResponse> {
  const params = new URLSearchParams({
    family: options.family,
    q: options.q,
    searchable_only: String(options.searchableOnly),
    limit: String(options.limit),
    offset: String(options.offset)
  });
  return getJson<DataFieldsResponse>(`/api/data/fields?${params}`, { signal: options.signal });
}

export async function fetchDataHealth(universe: string, signal?: AbortSignal): Promise<DataHealthResponse> {
  const params = new URLSearchParams({ universe });
  return getJson<DataHealthResponse>(`/api/data/health?${params}`, { signal });
}

export async function fetchRunCompare(options: {
  universe: string;
  leftRunId: string;
  rightRunId: string;
  topN: number;
  signal?: AbortSignal;
}): Promise<RunCompareResponse> {
  const params = new URLSearchParams({
    universe: options.universe,
    left_run_id: options.leftRunId,
    right_run_id: options.rightRunId,
    top_n: String(options.topN)
  });
  return getJson<RunCompareResponse>(`/api/runs/compare?${params}`, { signal: options.signal });
}

export async function fetchLibrary(universe = "", signal?: AbortSignal): Promise<LibraryResponse> {
  const params = new URLSearchParams();
  if (universe) {
    params.set("universe", universe);
  }
  const suffix = params.toString() ? `?${params}` : "";
  return getJson<LibraryResponse>(`/api/library${suffix}`, { signal });
}

export async function fetchFactorLibraryStatus(
  universe: string,
  runId: string,
  factor: string,
  signal?: AbortSignal
): Promise<LibraryStatusResponse> {
  const params = new URLSearchParams({ universe, run_id: runId });
  return getJson<LibraryStatusResponse>(`/api/factors/${encodeURIComponent(factor)}/library/status?${params}`, { signal });
}

export async function checkFactorLibrary(universe: string, runId: string, factor: string): Promise<LibraryCheckResponse> {
  return getJson<LibraryCheckResponse>(`/api/factors/${encodeURIComponent(factor)}/library/check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universe, run_id: runId })
  });
}

export async function submitFactorLibrary(
  universe: string,
  runId: string,
  factor: string,
  submittedBy = "dashboard"
): Promise<LibrarySubmitResponse> {
  return getJson<LibrarySubmitResponse>(`/api/factors/${encodeURIComponent(factor)}/library/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universe, run_id: runId, submitted_by: submittedBy })
  });
}

export async function fetchSuperalphaComponents(universe: string, signal?: AbortSignal): Promise<SuperalphaComponentsResponse> {
  const params = new URLSearchParams({ universe });
  return getJson<SuperalphaComponentsResponse>(`/api/superalphas/components?${params}`, { signal });
}

export async function fetchSuperalphaRuns(universe: string, signal?: AbortSignal): Promise<SuperalphaRunsResponse> {
  const params = new URLSearchParams({ universe });
  return getJson<SuperalphaRunsResponse>(`/api/superalphas/runs?${params}`, { signal });
}

export async function runSuperalphaBacktest(request: SuperalphaBacktestRequest): Promise<SuperalphaBacktestResponse> {
  return getJson<SuperalphaBacktestResponse>("/api/superalphas/backtest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
}

export async function renameSuperalphaRun(superalphaId: string, name: string): Promise<SuperalphaRenameResponse> {
  return getJson<SuperalphaRenameResponse>(`/api/superalphas/${encodeURIComponent(superalphaId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
}

export async function fetchSuperalphaDetail(
  superalphaId: string,
  signal?: AbortSignal,
  includeTest = false
): Promise<SuperalphaDetailResponse> {
  const params = new URLSearchParams({ include_test: String(includeTest) });
  return getJson<SuperalphaDetailResponse>(`/api/superalphas/${encodeURIComponent(superalphaId)}/detail?${params}`, { signal });
}

export async function fetchLiveStatus(universe: string, signal?: AbortSignal): Promise<LiveStatusResponse> {
  const params = new URLSearchParams({ universe });
  return getJson<LiveStatusResponse>(`/api/live/status?${params}`, { signal });
}

export async function fetchActiveLiveSuperalphas(universe: string, signal?: AbortSignal): Promise<LiveActiveResponse> {
  const params = new URLSearchParams({ universe, include_paused: "true" });
  return getJson<LiveActiveResponse>(`/api/live/superalphas/active?${params}`, { signal });
}

export async function activateLiveSuperalpha(universe: string, superalphaId: string): Promise<LiveActivateResponse> {
  return getJson<LiveActivateResponse>("/api/live/superalphas/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universe, superalpha_id: superalphaId })
  });
}

export async function updateLiveSuperalphaStatus(universe: string, superalphaId: string, status: string): Promise<{ status: string }> {
  return getJson<{ status: string }>(`/api/live/superalphas/active/${encodeURIComponent(superalphaId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universe, status })
  });
}

export async function fetchLiveDataStatus(universe: string, signal?: AbortSignal): Promise<LiveDataStatusResponse> {
  const params = new URLSearchParams({ universe });
  return getJson<LiveDataStatusResponse>(`/api/live/data-status?${params}`, { signal });
}

export async function fetchLiveHoldings(
  universe: string,
  superalphaId: string,
  limit = 200,
  signal?: AbortSignal
): Promise<LiveHoldingsResponse> {
  const params = new URLSearchParams({ universe, superalpha_id: superalphaId, limit: String(limit) });
  return getJson<LiveHoldingsResponse>(`/api/live/holdings?${params}`, { signal });
}

export async function fetchLiveOrders(
  universe: string,
  superalphaId: string,
  limit = 500,
  signal?: AbortSignal
): Promise<LiveOrdersResponse> {
  const params = new URLSearchParams({ universe, superalpha_id: superalphaId, limit: String(limit) });
  return getJson<LiveOrdersResponse>(`/api/live/orders?${params}`, { signal });
}

export async function fetchClosedLoopParams(signal?: AbortSignal): Promise<ClosedLoopParamsResponse> {
  return getJson<ClosedLoopParamsResponse>("/api/closed-loop/params", { signal });
}

export async function fetchClosedLoopJobs(signal?: AbortSignal): Promise<ClosedLoopJobsResponse> {
  return getJson<ClosedLoopJobsResponse>("/api/closed-loop/jobs", { signal });
}

export async function fetchClosedLoopJob(jobId: string, signal?: AbortSignal): Promise<ClosedLoopJob> {
  return getJson<ClosedLoopJob>(`/api/closed-loop/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export async function createClosedLoopJob(params: Record<string, unknown>): Promise<ClosedLoopJob> {
  return getJson<ClosedLoopJob>("/api/closed-loop/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params })
  });
}

export async function cancelClosedLoopJob(jobId: string): Promise<ClosedLoopJob> {
  return getJson<ClosedLoopJob>(`/api/closed-loop/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST"
  });
}
