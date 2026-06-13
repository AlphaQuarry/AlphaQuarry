export type UniverseSummary = {
  name: string;
  run_count: number;
  latest_created_at_utc: string;
};

export type FreshnessWarning = {
  code: string;
  severity: "info" | "warning" | "error" | string;
  universe?: string;
  message: string;
  days_since?: number | null;
};

export type DashboardOverviewResponse = {
  status: "ok" | "missing" | "error" | string;
  store_root: string;
  universes: UniverseSummary[];
  universe_count: number;
  run_count: number;
  latest_analysis_at_utc: string;
  field_catalog_status?: string | null;
  field_catalog_max_available_end?: string | null;
  field_catalog_row_count?: number | null;
  live_status_by_universe: Record<string, Record<string, unknown>>;
  freshness_warnings: FreshnessWarning[];
  generated_at_utc?: string;
};

export type PreflightResponse = {
  status: "ok" | "warn" | "error" | string;
  warnings: string[];
  infos: string[];
  remediations: string[];
  strict_exit_code: number;
};

export type AnalysisRun = {
  universe: string;
  run_id: string;
  label?: string;
  is_scoreboard?: boolean;
  period: number;
  layers: number;
  created_at_utc: string;
  analysis_dir: string;
  factor_count: number;
  has_dashboard_metrics: boolean;
  has_factor_metrics: boolean;
  has_portfolio_pnl: boolean;
  has_benchmark_pnl?: boolean;
  has_visualizations: boolean;
  has_phase_metrics?: boolean;
  has_ic_rows?: boolean;
  has_analysis_data?: boolean;
  available_phases?: string[];
  phase_config?: PhaseConfig | null;
  benchmark_config?: Record<string, unknown> | null;
  benchmark_status?: Record<string, unknown> | null;
};

export type FactorMetric = {
  factor: string;
  period: number;
  layers: number;
  expression: string;
  ic_mean: number | null;
  ir: number | null;
  long_short_total_return: number | null;
  long_short_annualized_return: number | null;
  long_short_volatility: number | null;
  long_short_sharpe_ratio: number | null;
  long_short_max_drawdown: number | null;
  long_short_fitness_ratio: number | null;
  best_layer_total_return: number | null;
  best_layer_annualized_return: number | null;
  best_layer_volatility: number | null;
  best_layer_sharpe: number | null;
  best_layer_max_drawdown: number | null;
  best_layer_fitness_ratio: number | null;
  best_minus_universe_annualized_return: number | null;
  benchmark_annualized_return?: number | null;
  long_short_excess_annualized_return_vs_benchmark?: number | null;
  long_only_excess_annualized_return_vs_benchmark?: number | null;
  best_minus_benchmark_annualized_return?: number | null;
  turnover_long_only_mean: number | null;
  margin_long_only: number | null;
  score_predictive_power: number | null;
  score_long_only_performance: number | null;
  score_stability: number | null;
  score_tradeability: number | null;
  score_total: number | null;
  ic_decay_spearman?: number | null;
  effectiveness_tier: string | null;
  analysis_run_id?: string | null;
  scoreboard_score?: number | null;
  train_score_total?: number | null;
  feedback_phase?: string | null;
  feedback_score?: number | null;
  [key: string]: string | number | boolean | null | undefined;
};

export type PnlRow = {
  factor: string;
  trade_date: string;
  portfolio: string;
  return: number | null;
  cum_return: number | null;
  return_gross?: number | null;
  cum_return_gross?: number | null;
  transaction_cost?: number | null;
  return_net?: number | null;
  cum_return_net?: number | null;
  has_net_pnl?: boolean | null;
  cost_model?: string | null;
  holding_count: number | null;
  turnover: number | null;
  buy_turnover?: number | null;
  sell_turnover?: number | null;
  blocked_buy_ratio: number | null;
  blocked_sell_ratio: number | null;
  tradability_return_drag: number | null;
  phase?: string | null;
};

export type PhaseWindow = {
  key: "train" | "val" | "test" | string;
  label: string;
  start: string;
  end?: string | null;
  available?: boolean;
  visible_default?: boolean;
};

export type PhaseConfig = {
  windows: PhaseWindow[];
  available_phases?: string[];
  feedback_phase?: string | null;
  test_default_visible?: boolean;
  phase_metric_min_obs?: number;
};

export type PhaseMetrics = Record<string, Record<string, number | string | null> | number | string | null>;

export type PortfolioMetricRow = {
  portfolio: string;
  label: string;
  total_return: number | null;
  annualized_return: number | null;
  excess_annualized_return: number | null;
  annualized_volatility: number | null;
  max_drawdown: number | null;
  turnover: number | null;
  sharpe: number | null;
  fitness: number | null;
  obs?: number | null;
};

export type PortfolioMetricsResponse = {
  scope_phase: string;
  rows: PortfolioMetricRow[];
  rows_net?: PortfolioMetricRow[];
  net_available?: boolean;
  benchmark_available?: boolean;
  message?: string | null;
};

export type PnlResponse = {
  status: "ok" | "missing" | "invalid" | "empty";
  factor: string;
  rows: PnlRow[];
  message?: string;
  benchmark_status?: Record<string, unknown> | null;
  phase_config?: PhaseConfig | null;
  phase_metrics?: PhaseMetrics | null;
  portfolio_metrics?: PortfolioMetricsResponse | null;
};

export type FactorVisualization = {
  plot_id: string;
  category: string;
  title: string;
  url: string;
  width?: number | null;
  height?: number | null;
  sort_order?: number | null;
};

export type VisualizationResponse = {
  status: "ok" | "missing" | "invalid" | "empty";
  factor: string;
  images: FactorVisualization[];
  message?: string | null;
};

export type AnalysisIcPoint = {
  trade_date: string;
  phase?: string | null;
  ic: number | null;
  cumulative_ic: number | null;
};

export type AnalysisYearlyIc = {
  phase: string;
  year: string;
  ic_mean: number | null;
  obs: number | null;
};

export type AnalysisMonthlyIc = {
  phase: string;
  month: string;
  ic_mean: number | null;
  obs: number | null;
};

export type AnalysisCoveragePoint = {
  trade_date: string;
  phase?: string | null;
  coverage_rate: number | null;
  non_missing_obs?: number | null;
  total_obs?: number | null;
};

export type AnalysisDistributionBin = {
  factor: string;
  phase: string;
  bin_index: number;
  bin_left: number | null;
  bin_right: number | null;
  bin_mid: number | null;
  count: number | null;
  total_count: number | null;
};

export type AnalysisIcDecayPoint = {
  factor: string;
  phase: string;
  lag: number;
  ic: number | null;
  half_life?: number | null;
  ic_decay_rank_corr?: number | null;
};

export type AnalysisLayerTerminalReturn = {
  phase: string;
  portfolio: string;
  layer: string;
  terminal_return: number | null;
  obs: number | null;
  rank_corr?: number | null;
};

export type AnalysisDataResponse = {
  status: "ok" | "missing" | "invalid" | "empty";
  factor: string;
  phase_config?: PhaseConfig | null;
  phase_metrics?: PhaseMetrics | null;
  ic_series: AnalysisIcPoint[];
  yearly_ic: AnalysisYearlyIc[];
  monthly_ic?: AnalysisMonthlyIc[];
  coverage_series?: AnalysisCoveragePoint[];
  distribution: AnalysisDistributionBin[];
  ic_distribution: AnalysisDistributionBin[];
  ic_decay: AnalysisIcDecayPoint[];
  layer_terminal_return: AnalysisLayerTerminalReturn[];
  message?: string | null;
};

export type SortDir = "asc" | "desc";

export type DataFamilySummary = {
  family: string;
  field_count: number;
  searchable_count: number;
  enabled_count: number;
  avg_coverage_rate: number | null;
  min_available_start?: string | null;
  max_available_end?: string | null;
  source_tables: string[];
};

export type DataField = {
  field_name: string;
  factor_family: string;
  category: string;
  source_table: string;
  field_type?: string | null;
  dtype: string;
  unit?: string | null;
  available_start?: string | null;
  available_end?: string | null;
  is_default_enabled?: boolean | null;
  is_searchable?: boolean | null;
  description?: string | null;
  field_role?: string | null;
  available_at?: string | null;
  preprocessing_policy?: string | null;
  leakage_safe?: boolean | null;
  coverage_rate?: number | null;
  finite_rate?: number | null;
  coverage_status?: string | null;
  coverage_updated_at_utc?: string | null;
};

export type DataFamiliesResponse = {
  status: "ok" | "missing";
  message?: string | null;
  metadata_note?: string | null;
  source?: string;
  duckdb_path?: string;
  row_count: number;
  families: DataFamilySummary[];
};

export type DataFieldsResponse = {
  status: "ok" | "missing";
  message?: string | null;
  metadata_note?: string | null;
  source?: string;
  duckdb_path?: string;
  row_count: number;
  total: number;
  fields: DataField[];
};

export type DataHealthWarning = {
  code: string;
  severity: string;
  message: string;
  days_since?: number | null;
};

export type DataHealthFamily = {
  family: string;
  field_count: number;
  searchable_count: number;
  avg_coverage_rate: number | null;
  low_coverage_count: number;
  max_available_end?: string | null;
};

export type DataHealthResponse = {
  status: "ok" | string;
  universe: string;
  catalog: Record<string, unknown>;
  families: DataHealthFamily[];
  universe_base: Record<string, unknown>;
  closed_loop_health: Record<string, unknown>;
  quality_artifact: Record<string, unknown>;
  thresholds?: Record<string, number>;
  warnings: DataHealthWarning[];
  generated_at_utc?: string;
};

export type LibraryFactor = {
  universe?: string;
  factor: string;
  expression?: string | null;
  analysis_run_id?: string | null;
  status?: string | null;
  score?: number | null;
  score_basis?: string | null;
  acceptance_mode?: string | null;
  submitted_by?: string | null;
  submitted_at_utc?: string | null;
  checked_at_utc?: string | null;
  reject_reasons?: string | null;
  rejection_reason?: string | null;
  library_status_reason?: string | null;
  signal_corr?: number | null;
  ic_corr?: number | null;
  long_only_corr?: number | null;
  long_short_corr?: number | null;
  max_signal_corr?: number | null;
  max_ic_corr?: number | null;
  max_pnl_corr?: number | null;
  max_any_corr?: number | null;
  nearest_factor_id?: string | null;
  nearest_expression_hash?: string | null;
  high_corr_peer_count?: number | null;
  candidate_long_only_sharpe?: number | null;
  candidate_long_short_sharpe?: number | null;
  max_peer_long_only_sharpe?: number | null;
  max_peer_long_short_sharpe?: number | null;
  long_only_sharpe_delta?: number | null;
  long_short_sharpe_delta?: number | null;
  override_portfolio?: string | null;
  override_reason?: string | null;
  library_status_effective?: string | null;
  legacy_status_warning?: string | null;
};

export type LibraryResponse = {
  status: "ok" | "missing";
  total: number;
  factors: LibraryFactor[];
  message?: string | null;
};

export type LibraryStatusResponse = {
  status: "ok" | "missing" | "error";
  factor: string;
  library_status: string;
  registry_row?: LibraryFactor | null;
  can_check?: boolean;
  can_submit?: boolean;
};

export type LibraryCheckDecision = "pass" | "pass_with_override" | "staging" | "reject";

export type LibraryCheckResponse = {
  status: "ok" | "missing" | "blocked" | "disabled" | "error";
  factor: string;
  universe?: string;
  analysis_run_id?: string;
  decision: LibraryCheckDecision;
  can_submit: boolean;
  score: number | null;
  score_basis: string | null;
  acceptance_mode?: string | null;
  reason?: string | null;
  corr?: Record<string, number | string | null>;
  override?: Record<string, number | string | boolean | null>;
  high_corr_peers?: Array<Record<string, unknown>>;
  peer_details?: Array<Record<string, unknown>>;
  row?: LibraryFactor | null;
  thresholds?: Record<string, number>;
};

export type LibrarySubmitResponse = {
  status: "ok" | "blocked" | "error";
  submitted: boolean;
  factor: string;
  library_status?: string;
  acceptance_mode?: string | null;
  registry_path?: string | null;
  row?: LibraryFactor | null;
  check?: LibraryCheckResponse | null;
  reason?: string | null;
};

export type SuperalphaComponent = LibraryFactor & {
  weight?: number | null;
  signal_artifact_path?: string | null;
  ic_artifact_path?: string | null;
  pnl_artifact_path?: string | null;
  signal_available?: boolean | null;
  signal_status_reason?: string | null;
  signal_status?: "compact" | "raw" | "cached" | "reproducible" | "reproduced" | "duckdb_fallback" | "read_error" | "unavailable";
  can_reproduce?: boolean;
  can_backtest?: boolean;
  signal_source?: string | null;
  reproduce_source_mode?: string | null;
  strict_reproducibility?: boolean;
  reproduce_warning?: string | null;
  cache_path?: string | null;
  direction_sign?: number | null;
  direction_status?: string | null;
};

export type SuperalphaComponentsResponse = {
  status: "ok" | "missing" | "error";
  universe: string;
  total: number;
  components: SuperalphaComponent[];
  message?: string | null;
};

export type SuperalphaRun = {
  superalpha_id: string;
  run_id?: string;
  name?: string;
  display_name?: string;
  universe: string;
  created_at_utc: string;
  combo_expression?: string | null;
  component_join?: string | null;
  component_count?: number | null;
  components?: SuperalphaComponent[];
  status?: string | null;
  summary?: Record<string, number | string | null>;
  artifact_path?: string | null;
  resource_summary?: Record<string, unknown>;
  cache_summary?: Record<string, unknown>;
  cleanup_summary?: Record<string, unknown>;
};

export type SuperalphaRunsResponse = {
  status: "ok" | "missing" | "error";
  universe: string;
  total: number;
  runs: SuperalphaRun[];
  message?: string | null;
};

export type SuperalphaBacktestRequest = {
  universe: string;
  factor_ids: string[];
  combo_expression: string;
  name?: string;
  rerun?: boolean;
  component_join?: "concat" | "inner";
  allow_reproduce_fallback?: boolean;
  max_components?: number;
  duckdb_memory_limit?: string;
  duckdb_max_temp_directory_size?: string;
  duckdb_threads?: string;
};

export type SuperalphaBacktestResponse = {
  status: "ok" | "cached" | "error";
  superalpha_id: string;
  summary?: Record<string, number | string | null>;
  artifact_path?: string | null;
  meta?: Record<string, unknown>;
  message?: string | null;
  cache_stage?: string | null;
  warnings?: string[];
  resource_diagnostics_path?: string | null;
  cleanup_summary?: Record<string, unknown>;
  cache_summary?: Record<string, unknown>;
  component_resolution_summary?: Array<Record<string, unknown>>;
  component_status_counts?: Record<string, number>;
};

export type SuperalphaDetailResponse = {
  status: "ok" | "missing" | "error";
  superalpha_id: string;
  universe: string;
  meta?: Record<string, unknown>;
  run: AnalysisRun;
  factor: FactorMetric;
  pnl: PnlResponse;
  analysis_data: AnalysisDataResponse;
  metrics?: FactorMetric | Record<string, unknown> | null;
  message?: string | null;
};

export type SuperalphaRenameResponse = {
  status: "ok" | "error";
  superalpha_id: string;
  run: SuperalphaRun;
  message?: string | null;
};

export type LiveSnapshot = {
  superalpha_id: string;
  universe: string;
  component_factor_ids?: string[];
  component_expressions?: string[];
  component_weights?: number[];
  combo_expression?: string;
  summary_metrics?: Record<string, number | string | null>;
  source_meta_path?: string;
  source_meta_hash?: string;
};

export type LiveSuperalpha = {
  superalpha_id: string;
  display_name?: string;
  status: "active" | "paused" | "retired" | string;
  activated_at_utc?: string;
  last_live_run_id?: string;
  last_signal_date?: string;
  last_execute_date?: string;
  source_meta_exists?: boolean;
  snapshot?: LiveSnapshot;
};

export type LiveActiveResponse = {
  status: "ok" | "missing" | "error";
  universe: string;
  total: number;
  superalphas: LiveSuperalpha[];
};

export type LiveStatusResponse = {
  status: "ok" | "missing" | "error";
  universe: string;
  updated_at_utc?: string;
  active_total?: number;
  superalphas: Array<Record<string, unknown>>;
  message?: string | null;
};

export type LiveDataStatusResponse = {
  status: "ready" | "data_not_ready" | "no_active_superalpha" | "missing" | "error" | string;
  universe: string;
  requested_date?: string;
  common_ready_date?: string;
  resolved_signal_date?: string;
  blocking_fields?: string[];
  fields?: Array<Record<string, unknown>>;
};

export type LiveHoldingsResponse = {
  status: "ok" | "empty" | "missing" | "error";
  universe: string;
  superalpha_id: string;
  latest?: Record<string, unknown>;
  rows: Array<Record<string, unknown>>;
};

export type LiveOrdersResponse = {
  status: "ok" | "empty" | "missing" | "error";
  universe: string;
  superalpha_id: string;
  latest?: Record<string, unknown> | null;
  account: Record<string, unknown>;
  summary: Record<string, unknown>;
  rows: Array<Record<string, unknown>>;
};

export type LiveActivateResponse = {
  status: "ok" | "error";
  record: LiveSuperalpha;
  active_count: number;
};

export type ClosedLoopParam = {
  name: string;
  label: string;
  type: "string" | "number" | "boolean" | "select" | "date" | "json" | string;
  default: string | number | boolean | null;
  required?: boolean;
  options?: string[];
  min?: number;
  max?: number;
  risk?: string;
  placeholder?: string;
};

export type ClosedLoopParamGroup = {
  id: string;
  label: string;
  params: ClosedLoopParam[];
};

export type ClosedLoopParamsResponse = {
  status: "ok" | string;
  groups: ClosedLoopParamGroup[];
  presets?: Array<{
    id: string;
    label: string;
    description: string;
    risk: string;
    params: Record<string, unknown>;
  }>;
  safe_defaults?: Record<string, unknown>;
};

export type ClosedLoopJob = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted" | string;
  pid?: number | null;
  universe?: string;
  created_at_utc?: string;
  started_at_utc?: string;
  ended_at_utc?: string;
  exit_code?: number | null;
  params?: Record<string, unknown>;
  command?: string[];
  job_dir?: string;
  stdout_tail?: string;
  stderr_tail?: string;
  stdout_bytes?: number;
  stderr_bytes?: number;
  result_summary?: Record<string, unknown>;
  external_process?: boolean;
  failure_category?: string;
  failure_title?: string;
  failure_hint?: string;
  lock_owner?: Record<string, unknown> | null;
  status_label?: string;
  status_hint?: string;
  lock_age_seconds?: number | null;
  lock_stale_hint?: string;
};

export type ClosedLoopJobsResponse = {
  status: "ok" | string;
  total: number;
  jobs: ClosedLoopJob[];
};

export type RunCompareResponse = {
  status: "ok" | string;
  universe: string;
  top_n: number;
  left: AnalysisRun;
  right: AnalysisRun;
  left_artifact_status?: string;
  right_artifact_status?: string;
  metrics: Record<string, { left: number | null; right: number | null; delta: number | null }>;
  overlap: {
    overlap_count: number;
    overlap_ratio: number;
    shared_factors: string[];
    left_only: string[];
    right_only: string[];
    left_top?: string[];
    right_top?: string[];
  };
  warnings: string[];
};
