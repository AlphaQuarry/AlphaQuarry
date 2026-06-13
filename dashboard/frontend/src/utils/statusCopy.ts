export const STATUS_COPY = {
  loading: {
    dataHealth: "Loading data health",
    fieldCatalog: "Loading field catalog"
  },
  empty: {
    closedLoopJobs: "No dashboard-launched closed-loop jobs yet.",
    runCompareNeedsTwoRuns: "At least two real analysis runs are required for comparison.",
    runCompareNeedsMetrics: "At least two runs with metrics artifacts are recommended for a meaningful comparison.",
    chooseDifferentRuns: "Choose two different real analysis runs.",
    noLogOutput: "No log output yet."
  },
  missing: {
    artifact: "Missing artifact",
    notRefreshed: "Not refreshed"
  },
  error: {
    api: "API error"
  }
} as const;
