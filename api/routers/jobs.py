"""Jobs CRUD endpoints.

MVP semantics:
  POST /jobs        — create job record + (optionally) populate `photos` rows
                      from S3 prefix. Does NOT launch a RunPod pod yet — that
                      step is manual via runner scripts for now.
  GET  /jobs        — list jobs, newest first
  GET  /jobs/{id}   — job details + progress
  POST /jobs/{id}/stop / /resume — flips status only (pod control deferred)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from api.auth import require_api_key
from api.db import get_db
from api.models import Job, JobStatus, Photo, PhotoStatus
from api.s3_client import list_prefix
from api.schemas import JobCreate, JobOut, JobProgress

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


def _job_to_out(j: Job) -> JobOut:
    return JobOut.model_validate(j)


@router.post("", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(body: JobCreate, db: Session = Depends(get_db)) -> JobOut:
    if not body.prefix and not body.keys:
        raise HTTPException(400, "Either 'prefix' or 'keys' must be provided")

    job = Job(
        id=str(uuid.uuid4()),
        prefix=body.prefix,
        mode=body.mode,
        status=JobStatus.queued,
        notes=body.notes,
    )
    db.add(job)
    db.flush()  # get id

    keys: list[str] = []
    if body.keys:
        keys = list(body.keys)
    elif body.prefix:
        try:
            keys = list(list_prefix(body.prefix))
        except RuntimeError as e:
            # S3 not configured — accept job but with total=0; user can ingest later
            logger.warning(f"S3 unavailable, job {job.id} created without photos: {e}")
        except Exception as e:  # noqa: BLE001
            db.rollback()
            raise HTTPException(502, f"S3 listing failed: {e}") from e

    if keys:
        db.add_all([
            Photo(job_id=job.id, s3_key=k, status=PhotoStatus.pending) for k in keys
        ])
        job.total = len(keys)

    db.commit()
    db.refresh(job)
    logger.info(f"Created job {job.id} mode={job.mode.value} total={job.total}")
    return _job_to_out(job)


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = 50, db: Session = Depends(get_db)) -> list[JobOut]:
    rows = db.execute(select(Job).order_by(desc(Job.created_at)).limit(limit)).scalars().all()
    return [_job_to_out(j) for j in rows]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_to_out(job)


@router.get("/{job_id}/progress", response_model=JobProgress)
def get_progress(job_id: str, db: Session = Depends(get_db)) -> JobProgress:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    pct = (job.processed / job.total * 100) if job.total else 0.0
    return JobProgress(
        id=job.id,
        status=job.status,
        total=job.total,
        processed=job.processed,
        review_count=job.review_count,
        error_count=job.error_count,
        percent=round(pct, 2),
    )


@router.post("/{job_id}/stop", response_model=JobOut)
def stop_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in {JobStatus.completed, JobStatus.failed}:
        raise HTTPException(409, f"Job already in terminal state: {job.status.value}")
    job.status = JobStatus.stopped
    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return _job_to_out(job)


@router.post("/{job_id}/resume", response_model=JobOut)
def resume_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in {JobStatus.stopped, JobStatus.paused, JobStatus.failed}:
        raise HTTPException(
            409, f"Job in status {job.status.value} cannot be resumed"
        )
    job.status = JobStatus.queued
    job.finished_at = None
    db.commit()
    db.refresh(job)
    return _job_to_out(job)
