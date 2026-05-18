"""ORM models for the orchestrator.

Three tables:
  jobs    — one batch processing request (a prefix or explicit list of keys)
  photos  — per-input-file record: hash, status, output_key, error
  pods    — one RunPod GPU instance launched for a job (1:N with jobs)
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"


class JobMode(str, enum.Enum):
    full = "full"
    resize = "resize"


class PhotoStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    review = "review"
    error = "error"


class PodStatus(str, enum.Enum):
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"
    terminated = "terminated"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prefix: Mapped[Optional[str]] = mapped_column(String(1024))
    mode: Mapped[JobMode] = mapped_column(Enum(JobMode), default=JobMode.resize)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.queued, index=True
    )
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    pods: Mapped[list["Pod"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    s3_key: Mapped[str] = mapped_column(String(1024), index=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    status: Mapped[PhotoStatus] = mapped_column(
        Enum(PhotoStatus), default=PhotoStatus.pending, index=True
    )
    output_key: Mapped[Optional[str]] = mapped_column(String(1024))
    review_reason: Mapped[Optional[str]] = mapped_column(String(64))
    error: Mapped[Optional[str]] = mapped_column(Text)
    processing_ms: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Pod(Base):
    __tablename__ = "pods"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True)
    gpu_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[PodStatus] = mapped_column(Enum(PodStatus), default=PodStatus.starting)
    cost_per_hour: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    runtime_seconds: Mapped[Optional[int]] = mapped_column(Integer)

    job: Mapped[Job] = relationship(back_populates="pods")
