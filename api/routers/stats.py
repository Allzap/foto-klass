"""Aggregate stats + cost endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.auth import require_api_key
from api.config import get_settings
from api.db import get_db
from api.models import Job, Photo, PhotoStatus, Pod, PodStatus
from api.schemas import CostSummary

router = APIRouter(prefix="/stats", tags=["stats"], dependencies=[Depends(require_api_key)])


@router.get("/cost", response_model=CostSummary)
def cost_summary(db: Session = Depends(get_db)) -> CostSummary:
    cfg = get_settings()

    jobs_total = db.execute(select(func.count(Job.id))).scalar_one()

    photos_processed = db.execute(
        select(func.count(Photo.id)).where(Photo.status == PhotoStatus.done)
    ).scalar_one()

    # GPU time accumulated from completed pods only
    seconds = db.execute(
        select(func.coalesce(func.sum(Pod.runtime_seconds), 0)).where(
            Pod.status == PodStatus.completed
        )
    ).scalar_one()
    gpu_hours = float(seconds) / 3600.0
    gpu_cost = gpu_hours * cfg.runpod_rtx4090_per_hour

    # Rough storage estimate: 200 KB per output
    storage_gb = photos_processed * 200_000 / (1024**3)
    storage_cost = storage_gb * cfg.hetzner_storage_per_gb

    return CostSummary(
        jobs_total=jobs_total,
        photos_processed=photos_processed,
        gpu_hours=round(gpu_hours, 2),
        gpu_cost_usd=round(gpu_cost, 2),
        storage_estimate_gb=round(storage_gb, 2),
        storage_cost_per_month_usd=round(storage_cost, 2),
        total_to_date_usd=round(gpu_cost + storage_cost, 2),
    )
