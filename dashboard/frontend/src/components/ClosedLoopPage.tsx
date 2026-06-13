import { useEffect, useMemo, useRef, useState } from "react";
import { Play, RefreshCw, RotateCcw, Square } from "lucide-react";

import {
  cancelClosedLoopJob,
  createClosedLoopJob,
  fetchClosedLoopJob,
  fetchClosedLoopJobs,
  fetchClosedLoopParams
} from "../api";
import type { ClosedLoopJob, ClosedLoopParam, ClosedLoopParamGroup, ClosedLoopParamsResponse, UniverseSummary } from "../types";
import { errorMessage, isAbortError } from "../utils/format";
import { STATUS_COPY } from "../utils/statusCopy";

export function ClosedLoopPage({
  universe,
  universes,
  onUniverseChange
}: {
  universe: string;
  universes: UniverseSummary[];
  onUniverseChange: (value: string) => void;
}) {
  const [schema, setSchema] = useState<ClosedLoopParamsResponse | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [jobs, setJobs] = useState<ClosedLoopJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedJob, setSelectedJob] = useState<ClosedLoopJob | null>(null);
  const [selectedPreset, setSelectedPreset] = useState("custom");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const jobListRef = useRef<HTMLDivElement | null>(null);
  const prevJobCountRef = useRef(0);

  useEffect(() => {
    if (jobs.length > prevJobCountRef.current && jobListRef.current) {
      jobListRef.current.scrollTop = jobListRef.current.scrollHeight;
    }
    prevJobCountRef.current = jobs.length;
  }, [jobs.length]);

  useEffect(() => {
    const controller = new AbortController();
    fetchClosedLoopParams(controller.signal)
      .then((payload) => {
        setSchema(payload);
        setValues((current) => ({ ...defaultsFromSchema(payload.groups), ...current, universe: universe || current.universe || "cn_all" }));
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (universe) {
      setValues((current) => ({ ...current, universe }));
    }
  }, [universe]);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetchClosedLoopJobs(controller.signal)
      .then((payload) => {
        setJobs(payload.jobs);
        setSelectedJobId((current) => current || payload.jobs[0]?.job_id || "");
      })
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJob(null);
      return;
    }
    const controller = new AbortController();
    setError("");
    fetchClosedLoopJob(selectedJobId, controller.signal)
      .then(setSelectedJob)
      .catch((exc: unknown) => {
        if (!isAbortError(exc)) setError(errorMessage(exc));
      });
    return () => controller.abort();
  }, [selectedJobId, refreshKey]);

  useEffect(() => {
    if (!jobs.some((job) => job.status === "running") && selectedJob?.status !== "running") {
      return;
    }
    const timer = window.setInterval(() => setRefreshKey((value) => value + 1), 3000);
    return () => window.clearInterval(timer);
  }, [jobs, selectedJob]);

  const riskMessages = useMemo(() => riskHints(values), [values]);

  const applyPreset = (presetId: string) => {
    setSelectedPreset(presetId);
    if (presetId === "custom") return;
    const preset = schema?.presets?.find((row) => row.id === presetId);
    if (!preset) return;
    setValues((current) => ({ ...current, ...preset.params, universe: current.universe || universe || "cn_all" }));
  };

  const runJob = () => {
    const errors = validateParams(values);
    if (errors.length) {
      setError(errors.join("; "));
      return;
    }
    setSubmitting(true);
    setError("");
    createClosedLoopJob(values)
      .then((job) => {
        setSelectedJobId(job.job_id);
        setSelectedJob(job);
        setRefreshKey((value) => value + 1);
      })
      .catch((exc: unknown) => setError(errorMessage(exc)))
      .finally(() => setSubmitting(false));
  };

  const cancelJob = () => {
    if (!selectedJobId) return;
    if (!window.confirm("Cancel this running job?")) return;
    cancelClosedLoopJob(selectedJobId)
      .then((job) => {
        setSelectedJob(job);
        setRefreshKey((value) => value + 1);
      })
      .catch((exc: unknown) => setError(errorMessage(exc)));
  };

  const resetToDefaults = () => {
    if (schema) {
      setValues(defaultsFromSchema(schema.groups));
      setSelectedPreset("custom");
    }
  };

  return (
    <section className="data-page closed-loop-page">
      <header className="data-header">
        <div>
          <strong>Closed Loop Runner</strong>
          <span>Launch factor mining as an isolated local background job</span>
        </div>
        <dl>
          <div>
            <dt>Universe</dt>
            <dd>{String(values.universe || universe || "-")}</dd>
          </div>
          <div>
            <dt>Jobs</dt>
            <dd>{jobs.length}</dd>
          </div>
          <div>
            <dt>Selected</dt>
            <dd>{selectedJob?.status || "-"}</dd>
          </div>
        </dl>
      </header>

      {error ? <div className="data-error">{STATUS_COPY.error.api}: {error}</div> : null}

      <section className="field-panel">
        <div className="field-toolbar">
          <label>
            Universe
            <select
              value={String(values.universe || universe || "")}
              onChange={(event) => {
                onUniverseChange(event.target.value);
                setValues((current) => ({ ...current, universe: event.target.value }));
              }}
            >
              {universes.map((row) => (
                <option key={row.name} value={row.name}>{row.name}</option>
              ))}
              {!universes.length ? <option value="cn_all">cn_all</option> : null}
            </select>
          </label>
          <label>
            Preset
            <select value={selectedPreset} onChange={(event) => applyPreset(event.target.value)}>
              <option value="custom">Custom</option>
              {(schema?.presets ?? []).map((preset) => (
                <option key={preset.id} value={preset.id}>{preset.label}</option>
              ))}
            </select>
          </label>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button type="button" className="compact-button primary-action" onClick={runJob} disabled={submitting}>
              <Play size={14} /> {submitting ? "Starting..." : "Start job"}
            </button>
            <button type="button" className="compact-button" onClick={() => setRefreshKey((value) => value + 1)}>
              <RefreshCw size={14} /> Refresh
            </button>
            <button type="button" className="compact-button" onClick={resetToDefaults}>
              <RotateCcw size={14} /> Reset
            </button>
          </div>
        </div>
      </section>

      {riskMessages.length ? (
        <section className="overview-message-list standalone">
          {riskMessages.map((message) => <p key={message}>{message}</p>)}
        </section>
      ) : null}

      {selectedPreset !== "custom" ? (
        <section className="overview-message-list standalone">
          <p>{schema?.presets?.find((row) => row.id === selectedPreset)?.description}</p>
          <p>{presetSummary(schema?.presets?.find((row) => row.id === selectedPreset)?.params)}</p>
        </section>
      ) : null}

      <section className="closed-loop-layout">
        <section className="field-panel closed-loop-form">
          {(schema?.groups ?? []).map((group) => (
            <ParamGroup key={group.id} group={group} values={values} onChange={setValues} />
          ))}
        </section>

        <section className="field-panel">
          <div className="superalpha-panel-title">
            <strong>Jobs</strong>
            <span>{jobs.length} recent</span>
          </div>
          <div className="closed-loop-job-list" ref={jobListRef}>
            {!jobs.length ? <div className="metrics-empty">{STATUS_COPY.empty.closedLoopJobs}</div> : null}
            {jobs.map((job) => (
              <button
                key={job.job_id}
                type="button"
                className={selectedJobId === job.job_id ? "closed-loop-job-card active" : "closed-loop-job-card"}
                onClick={() => setSelectedJobId(job.job_id)}
              >
                <strong>{job.universe || "-"}</strong>
                <span className={`status-badge status-${job.status}`}>{job.status_label || job.status}</span>
                <small>{job.created_at_utc?.slice(0, 19).replace("T", " ") || job.job_id}</small>
              </button>
            ))}
          </div>

          <div className="superalpha-panel-title">
            <strong>Job Detail</strong>
            <span>{selectedJob?.job_id || "none"}</span>
          </div>
          <div className="live-state-list">
            <div><span>Status</span><strong>{selectedJob?.status_label || selectedJob?.status || "-"}</strong></div>
            <div><span>PID</span><strong>{selectedJob?.pid ?? "-"}</strong></div>
            <div><span>Exit</span><strong>{selectedJob?.exit_code ?? "-"}</strong></div>
            <div><span>External</span><strong>{selectedJob?.external_process ? "yes" : "no"}</strong></div>
            <div><span>Stdout</span><strong>{selectedJob?.stdout_bytes ?? 0} bytes</strong></div>
            <div><span>Stderr</span><strong>{selectedJob?.stderr_bytes ?? 0} bytes</strong></div>
          </div>
          {selectedJob?.status_hint ? (
            <section className="job-diagnosis subtle">
              <strong>Current State</strong>
              <p>{selectedJob.status_hint}</p>
            </section>
          ) : null}
          {selectedJob?.failure_category ? (
            <section className="job-diagnosis">
              <strong>{selectedJob.failure_title || selectedJob.failure_category}</strong>
              <p>{selectedJob.failure_hint}</p>
            </section>
          ) : null}
          {selectedJob?.lock_owner ? (
            <section className="job-diagnosis subtle">
              <strong>Lock Owner</strong>
              <p>
                PID {String(selectedJob.lock_owner.pid ?? "-")} on {String(selectedJob.lock_owner.hostname ?? "-")} - heartbeat{" "}
                {String(selectedJob.lock_owner.heartbeat_at_utc ?? "-")} - age {formatAge(selectedJob.lock_age_seconds)}
              </p>
              {selectedJob.lock_stale_hint ? <p>{selectedJob.lock_stale_hint}</p> : null}
            </section>
          ) : null}
          <div className="live-tabbar">
            <button type="button" className="compact-button" disabled={selectedJob?.status !== "running"} onClick={cancelJob}>
              <Square size={13} /> Cancel
            </button>
          </div>
          <pre className="job-log">{selectedJob?.stderr_tail || selectedJob?.stdout_tail || STATUS_COPY.empty.noLogOutput}</pre>
        </section>
      </section>
    </section>
  );
}

function ParamGroup({
  group,
  values,
  onChange
}: {
  group: ClosedLoopParamGroup;
  values: Record<string, unknown>;
  onChange: (updater: (current: Record<string, unknown>) => Record<string, unknown>) => void;
}) {
  const [open, setOpen] = useState(group.id === "basic");
  return (
    <section className="param-group">
      <button type="button" className="param-group-head" onClick={() => setOpen((value) => !value)}>
        <strong>{group.label}</strong>
        <span>{open ? "Hide" : "Show"}</span>
      </button>
      {open ? (
        <div className="param-grid">
          {group.params.map((param) => (
            <ParamField key={param.name} param={param} value={values[param.name] ?? param.default} onChange={(value) => onChange((current) => ({ ...current, [param.name]: value }))} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

function ParamField({ param, value, onChange }: { param: ClosedLoopParam; value: unknown; onChange: (value: unknown) => void }) {
  const isDefault = value === param.default || (value == null && param.default == null) || String(value ?? "") === String(param.default ?? "");
  const fieldClass = isDefault ? "param-field param-field-default" : "param-field";

  if (param.type === "boolean") {
    return (
      <label className={fieldClass}>
        <span>{param.label}</span>
        <select value={value ? "ON" : "OFF"} onChange={(event) => onChange(event.target.value === "ON")}>
          <option value="ON">ON</option>
          <option value="OFF">OFF</option>
        </select>
      </label>
    );
  }
  if (param.type === "select") {
    return (
      <label className={fieldClass}>
        <span>{param.label}</span>
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {(param.options ?? []).map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className={fieldClass}>
      <span>{param.label}</span>
      <input
        type={param.type === "number" ? "number" : param.type === "date" ? "date" : "text"}
        min={param.min}
        max={param.max}
        placeholder={param.placeholder || (param.default != null && param.default !== "" ? String(param.default) : undefined)}
        value={String(value ?? "")}
        onChange={(event) => {
          const raw = event.target.value;
          if (raw === "") {
            onChange(param.default);
          } else {
            onChange(param.type === "number" ? Number(raw) : raw);
          }
        }}
      />
    </label>
  );
}

function defaultsFromSchema(groups: ClosedLoopParamGroup[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const group of groups) {
    for (const param of group.params) {
      out[param.name] = param.default;
    }
  }
  return out;
}

function riskHints(values: Record<string, unknown>): string[] {
  const hints: string[] = [];
  if (!values.source_chunk_loading) hints.push("Source chunk loading is disabled; memory peaks can be much higher.");
  if (Number(values.max_eval) > 500) hints.push("Large max_eval values can materially increase runtime and disk artifacts.");
  if (values.include_visualization_png) hints.push("PNG visualization artifacts can increase runtime and disk usage.");
  if (!values.candidate_artifact_retention_enabled) hints.push("Candidate retention is disabled; old artifacts may accumulate.");
  return hints;
}

function presetSummary(params?: Record<string, unknown>): string {
  if (!params) return "";
  return Object.entries(params)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
}

function validateParams(values: Record<string, unknown>): string[] {
  const errors: string[] = [];
  const requestNew = Number(values.request_new);
  const batchSize = Number(values.batch_size);
  const maxEval = Number(values.max_eval);
  const iterations = Number(values.iterations);
  const hardLimit = Number(values.source_chunk_mem_hard_limit_mb);
  if (requestNew < 1) errors.push("Request new must be at least 1");
  if (batchSize < 1) errors.push("Batch size must be at least 1");
  if (batchSize > requestNew) errors.push("Batch size must be ≤ Request new");
  if (maxEval < 1 || maxEval > 5000) errors.push("Max evaluations must be 1-5000");
  if (iterations < 1 || iterations > 20) errors.push("Iterations must be 1-20");
  if (hardLimit <= 0) errors.push("Chunk memory hard limit must be non-zero");
  for (const key of ["score_weights_json", "layer_budget_json", "run_filters_json"]) {
    const raw = String(values[key] ?? "").trim();
    if (raw) {
      try { JSON.parse(raw); } catch { errors.push(`${key}: invalid JSON`); }
    }
  }
  return errors;
}

function formatAge(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "-";
  const value = Number(seconds);
  if (value < 60) return `${Math.round(value)}s`;
  if (value < 3600) return `${Math.round(value / 60)}m`;
  return `${(value / 3600).toFixed(1)}h`;
}
