from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import DashboardStore
from .closed_loop_jobs import (
    ClosedLoopJobConflict,
    cancel_closed_loop_job,
    create_closed_loop_job,
    get_closed_loop_job,
    list_closed_loop_jobs,
)
from .closed_loop_params import closed_loop_param_schema
from .preflight import run_preflight_checks
from alpha_mining.workflow.superalpha import SuperalphaBusyError


def create_app(
    store_root: str | Path = "data/alpha_universe_store",
    frontend_dist: str | Path | None = None,
) -> FastAPI:
    store = DashboardStore(store_root=store_root)
    app = FastAPI(title="Factor Dashboard", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return store.health()

    @app.get("/api/dashboard/overview")
    def dashboard_overview() -> dict:
        return store.overview()

    @app.get("/api/preflight")
    def preflight(config: str = "configs/datasource.local.yaml") -> dict:
        return run_preflight_checks(config=config, root=Path.cwd())

    @app.get("/api/closed-loop/params")
    def closed_loop_params() -> dict:
        return closed_loop_param_schema()

    @app.get("/api/closed-loop/jobs")
    def closed_loop_jobs(limit: int = Query(50, ge=1, le=200)) -> dict:
        return list_closed_loop_jobs(store_root=store.store_root, limit=limit)

    @app.post("/api/closed-loop/jobs")
    def closed_loop_job_create(body: dict) -> dict:
        try:
            return create_closed_loop_job(store_root=store.store_root, body=body, project_root=Path.cwd())
        except ClosedLoopJobConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "running_job": exc.running_job},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/closed-loop/jobs/{job_id}")
    def closed_loop_job_detail(job_id: str) -> dict:
        try:
            return get_closed_loop_job(store_root=store.store_root, job_id=job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/closed-loop/jobs/{job_id}/cancel")
    def closed_loop_job_cancel(job_id: str) -> dict:
        try:
            return cancel_closed_loop_job(store_root=store.store_root, job_id=job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/universes")
    def universes(refresh: bool = False) -> dict:
        if refresh:
            store.clear_cache()
        return {"universes": store.list_universes()}

    @app.get("/api/runs")
    def runs(universe: str = Query(..., min_length=1), refresh: bool = False) -> dict:
        if refresh:
            store.clear_cache()
        return {"runs": store.list_runs(universe=universe)}

    @app.get("/api/runs/compare")
    def run_compare(
        universe: str = Query(..., min_length=1),
        left_run_id: str = Query(..., min_length=1),
        right_run_id: str = Query(..., min_length=1),
        top_n: int = Query(50, ge=1, le=500),
    ) -> dict:
        try:
            return store.compare_runs(
                universe=universe,
                left_run_id=left_run_id,
                right_run_id=right_run_id,
                top_n=top_n,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors")
    def factors(
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
        q: str = "",
        sort_by: str = "feedback_score",
        sort_dir: str = "desc",
        effective_only: bool = False,
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> dict:
        try:
            return store.get_factors(
                universe=universe,
                run_id=run_id,
                q=q,
                sort_by=sort_by,
                sort_dir=sort_dir,
                effective_only=effective_only,
                limit=limit,
                offset=offset,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/data/families")
    def data_families() -> dict:
        return store.list_data_families()

    @app.get("/api/data/fields")
    def data_fields(
        family: str = "",
        q: str = "",
        searchable_only: bool = False,
        limit: int = Query(200, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> dict:
        return store.list_data_fields(
            family=family,
            q=q,
            searchable_only=searchable_only,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/data/health")
    def data_health(universe: str = Query(..., min_length=1)) -> dict:
        return store.data_health(universe=universe)

    @app.get("/api/library")
    def factor_library(universe: str = "", status: str = "", q: str = "") -> dict:
        payload = store.list_factor_library(universe=universe)
        rows = payload.get("factors", [])
        status_text = str(status or "").strip().lower()
        query = str(q or "").strip().lower()
        if status_text and status_text != "all":
            rows = [row for row in rows if str(row.get("status", "")).lower() == status_text]
        if query:
            rows = [
                row
                for row in rows
                if query
                in " ".join(
                    str(row.get(key, "") or "").lower()
                    for key in [
                        "factor",
                        "expression",
                        "analysis_run_id",
                        "nearest_factor_id",
                        "library_status_reason",
                        "rejection_reason",
                    ]
                )
            ]
        payload["factors"] = rows
        payload["total"] = len(rows)
        return payload

    @app.get("/api/factors/{factor}/library/status")
    def factor_library_status(
        factor: str,
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
    ) -> dict:
        try:
            return store.get_factor_library_status(universe=universe, run_id=run_id, factor=factor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/factors/{factor}/library/check")
    def factor_library_check(factor: str, body: dict) -> dict:
        try:
            return store.check_factor_library_candidate(
                universe=str(body.get("universe") or ""),
                run_id=str(body.get("run_id") or ""),
                factor=factor,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/factors/{factor}/library/submit")
    def factor_library_submit(factor: str, body: dict) -> dict:
        try:
            return store.submit_factor_library_candidate(
                universe=str(body.get("universe") or ""),
                run_id=str(body.get("run_id") or ""),
                factor=factor,
                submitted_by=str(body.get("submitted_by") or "dashboard"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/superalphas/components")
    def superalpha_components(universe: str = Query(..., min_length=1)) -> dict:
        try:
            return store.list_superalpha_components(universe=universe)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/superalphas/runs")
    def superalpha_runs(universe: str = Query(..., min_length=1)) -> dict:
        return store.list_superalpha_runs(universe=universe)

    @app.post("/api/superalphas/backtest")
    def superalpha_backtest(body: dict) -> dict:
        try:
            return store.run_superalpha_backtest(body)
        except SuperalphaBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/superalphas/{superalpha_id}")
    def superalpha_summary(superalpha_id: str) -> dict:
        try:
            return store.get_superalpha(superalpha_id=superalpha_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/superalphas/{superalpha_id}")
    def superalpha_update(superalpha_id: str, body: dict) -> dict:
        try:
            return store.rename_superalpha(superalpha_id=superalpha_id, name=str(body.get("name") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/superalphas/{superalpha_id}/detail")
    def superalpha_detail(superalpha_id: str, include_test: bool = False) -> dict:
        try:
            return store.get_superalpha_detail(superalpha_id=superalpha_id, include_test=include_test)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/live/status")
    def live_status(universe: str = Query(..., min_length=1)) -> dict:
        return store.get_live_status(universe=universe)

    @app.get("/api/live/superalphas/active")
    def live_active_superalphas(
        universe: str = Query(..., min_length=1),
        include_paused: bool = True,
        include_retired: bool = False,
    ) -> dict:
        return store.list_live_superalphas(
            universe=universe,
            include_paused=include_paused,
            include_retired=include_retired,
        )

    @app.post("/api/live/superalphas/active")
    def live_activate_superalpha(body: dict) -> dict:
        try:
            return store.activate_live_superalpha(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/live/superalphas/active/{superalpha_id}")
    def live_update_superalpha(superalpha_id: str, body: dict) -> dict:
        try:
            return store.update_live_superalpha_status(
                universe=str(body.get("universe") or ""),
                superalpha_id=superalpha_id,
                status=str(body.get("status") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/live/run")
    def live_run(body: dict) -> dict:
        try:
            return store.run_live_preflight(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/live/runs")
    def live_runs(universe: str = Query(..., min_length=1), superalpha_id: str = "") -> dict:
        return store.list_live_runs(universe=universe, superalpha_id=superalpha_id)

    @app.get("/api/live/data-status")
    def live_data_status(universe: str = Query(..., min_length=1)) -> dict:
        return store.get_live_data_status(universe=universe)

    @app.get("/api/live/holdings")
    def live_holdings(
        universe: str = Query(..., min_length=1),
        superalpha_id: str = Query(..., min_length=1),
        limit: int = Query(200, ge=1, le=5000),
    ) -> dict:
        return store.get_live_holdings(universe=universe, superalpha_id=superalpha_id, limit=limit)

    @app.get("/api/live/orders")
    def live_orders(
        universe: str = Query(..., min_length=1),
        superalpha_id: str = Query(..., min_length=1),
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict:
        return store.get_live_orders(universe=universe, superalpha_id=superalpha_id, limit=limit)

    @app.get("/api/factors/{factor}/pnl")
    def factor_pnl(
        factor: str,
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
        include_test: bool = False,
    ) -> dict:
        try:
            return store.get_factor_pnl(
                universe=universe,
                run_id=run_id,
                factor=factor,
                include_test=include_test,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor}/analysis-data")
    def factor_analysis_data(
        factor: str,
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
        include_test: bool = False,
    ) -> dict:
        try:
            return store.get_factor_analysis_data(
                universe=universe,
                run_id=run_id,
                factor=factor,
                include_test=include_test,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor}/visualizations")
    def factor_visualizations(
        factor: str,
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
    ) -> dict:
        try:
            return store.get_factor_visualizations(universe=universe, run_id=run_id, factor=factor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor}/visualizations/{plot_id}/image")
    def factor_visualization_image(
        factor: str,
        plot_id: str,
        universe: str = Query(..., min_length=1),
        run_id: str = Query(..., min_length=1),
    ) -> FileResponse:
        try:
            return FileResponse(
                store.get_visualization_image(universe=universe, run_id=run_id, factor=factor, plot_id=plot_id),
                media_type="image/png",
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    dist_path = Path(frontend_dist) if frontend_dist is not None else Path("dashboard/frontend/dist")
    if dist_path.exists():
        assets_path = dist_path / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = dist_path / path
            if path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist_path / "index.html")

    return app


def create_env_app() -> FastAPI:
    return create_app(
        store_root=os.environ.get("FACTOR_DASHBOARD_STORE_ROOT", "data/alpha_universe_store"),
        frontend_dist=os.environ.get("FACTOR_DASHBOARD_FRONTEND_DIST") or None,
    )


app = create_env_app()
