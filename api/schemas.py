"""Pydantic request / response schemas for the API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from api.models import JobMode, JobStatus, PodStatus


class JobCreate(BaseModel):
    prefix: Optional[str] = Field(
        None, description="S3 prefix to scan, e.g. 'inputs/parts/'"
    )
    keys: Optional[list[str]] = Field(
        None, description="Explicit list of S3 keys (alternative to prefix)"
    )
    mode: JobMode = JobMode.resize
    notes: Optional[str] = None


class JobOut(BaseModel):
    id: str
    prefix: Optional[str]
    mode: JobMode
    status: JobStatus
    total: int
    processed: int
    review_count: int
    error_count: int
    notes: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class JobProgress(BaseModel):
    id: str
    status: JobStatus
    total: int
    processed: int
    review_count: int
    error_count: int
    percent: float
    eta_seconds: Optional[int] = None
    avg_processing_ms: Optional[float] = None


class PodOut(BaseModel):
    id: str
    runpod_pod_id: Optional[str]
    gpu_type: str
    status: PodStatus
    cost_per_hour: float
    started_at: datetime
    finished_at: Optional[datetime]
    runtime_seconds: Optional[int]

    class Config:
        from_attributes = True


class CostSummary(BaseModel):
    jobs_total: int
    photos_processed: int
    gpu_hours: float
    gpu_cost_usd: float
    storage_estimate_gb: float
    storage_cost_per_month_usd: float
    total_to_date_usd: float


class HealthOut(BaseModel):
    status: str
    db: str
    runpod: str
    s3: str
