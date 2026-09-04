"""FastAPI backend for the IS2RE demo.

Serves test structures, per-structure predictions from the three checkpoints, and
the aggregate results table. Models and data are loaded once at startup and held in
memory. The frontend is served as static files from ``app/static``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    StructureDetail,
    StructureSummary,
)

log = logging.getLogger("is2re.demo")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            from app.service import build_service

            log.info("Building demo service (this loads models, may take a while) ...")
            app.state.service = build_service()
            log.info("Demo service ready.")
        yield

    app = FastAPI(title="IS2RE Demo", version="1.0.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health", response_model=HealthResponse)
    def health():
        if app.state.service is None:
            return {"status": "starting"}
        return {"status": "ok"}

    @app.get("/structures", response_model=list[StructureSummary])
    def list_structures():
        return app.state.service.list_structures()

    @app.get("/structures/{sid}", response_model=StructureDetail)
    def structure_detail(sid: int):
        detail = app.state.service.get_structure(sid)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Structure {sid} not found")
        return detail

    @app.get("/structures/{sid}/predictions", response_model=PredictionResponse)
    def structure_predictions(sid: int):
        predictions = app.state.service.get_predictions(sid)
        if predictions is None:
            raise HTTPException(status_code=404, detail=f"Structure {sid} not found")
        return predictions

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        return {"variants": app.state.service.results_table()}

    # Static frontend (registered last so API routes take precedence).
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()