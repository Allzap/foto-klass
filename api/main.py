"""FastAPI app entry point."""
from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sqlalchemy import text

from api.config import get_settings
from api.db import engine
from api.routers import jobs, stats
from api.runpod_client import RunPodClient
from api.s3_client import ping as s3_ping
from api.schemas import HealthOut

# Structured logging to stdout
logger.remove()
logger.add(sys.stdout, serialize=False, level="INFO",
           format="<green>{time:HH:mm:ss}</green> <level>{level}</level> {message}")


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="foto-klass orchestrator",
        version="0.2.0",
        description="REST API for managing batch photo-processing jobs on RunPod.",
    )

    # Open CORS for now — restrict in production once we know callers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(jobs.router)
    app.include_router(stats.router)

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        # DB
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            db_status = "ok"
        except Exception as e:  # noqa: BLE001
            db_status = f"down: {e.__class__.__name__}"

        # RunPod
        runpod_status = "ok" if RunPodClient().ping() else "down"

        # S3
        s3_status = "ok" if s3_ping() else ("not_configured" if not cfg.s3_endpoint else "down")

        overall = "ok" if (db_status == "ok" and runpod_status == "ok") else "degraded"
        return HealthOut(status=overall, db=db_status, runpod=runpod_status, s3=s3_status)

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service": "foto-klass orchestrator",
            "version": "0.2.0",
            "docs": "/docs",
            "health": "/health",
        }

    logger.info(f"API initialised. DB={cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}")
    return app


app = create_app()
