"""initial schema — jobs, photos, pods

Revision ID: 0001
Revises:
Create Date: 2026-05-18
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JOB_STATUS = ("queued", "running", "paused", "completed", "failed", "stopped")
JOB_MODE = ("full", "resize")
PHOTO_STATUS = ("pending", "processing", "done", "review", "error")
POD_STATUS = ("starting", "running", "completed", "failed", "terminated")


def upgrade() -> None:
    op.execute("CREATE TYPE jobstatus AS ENUM " + str(JOB_STATUS))
    op.execute("CREATE TYPE jobmode AS ENUM " + str(JOB_MODE))
    op.execute("CREATE TYPE photostatus AS ENUM " + str(PHOTO_STATUS))
    op.execute("CREATE TYPE podstatus AS ENUM " + str(POD_STATUS))

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("prefix", sa.String(length=1024)),
        sa.Column("mode", sa.Enum(*JOB_MODE, name="jobmode", create_type=False),
                  nullable=False, server_default="resize"),
        sa.Column("status", sa.Enum(*JOB_STATUS, name="jobstatus", create_type=False),
                  nullable=False, server_default="queued"),
        sa.Column("total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("processed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("review_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "photos",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(length=36), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64)),
        sa.Column("status", sa.Enum(*PHOTO_STATUS, name="photostatus", create_type=False),
                  nullable=False, server_default="pending"),
        sa.Column("output_key", sa.String(length=1024)),
        sa.Column("review_reason", sa.String(length=64)),
        sa.Column("error", sa.Text),
        sa.Column("processing_ms", sa.Integer),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_photos_job_id", "photos", ["job_id"])
    op.create_index("ix_photos_s3_key", "photos", ["s3_key"])
    op.create_index("ix_photos_sha256", "photos", ["sha256"])
    op.create_index("ix_photos_status", "photos", ["status"])

    op.create_table(
        "pods",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("job_id", sa.String(length=36), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("runpod_pod_id", sa.String(length=64), unique=True),
        sa.Column("gpu_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.Enum(*POD_STATUS, name="podstatus", create_type=False),
                  nullable=False, server_default="starting"),
        sa.Column("cost_per_hour", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("runtime_seconds", sa.Integer),
    )
    op.create_index("ix_pods_job_id", "pods", ["job_id"])


def downgrade() -> None:
    op.drop_table("pods")
    op.drop_table("photos")
    op.drop_table("jobs")
    for t in ("podstatus", "photostatus", "jobmode", "jobstatus"):
        op.execute(f"DROP TYPE IF EXISTS {t}")
