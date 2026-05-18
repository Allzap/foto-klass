"""Thin S3 client for Hetzner Object Storage.

Used for listing job inputs and writing manifests. Pod-side reads/writes
directly via boto3 too (separately).
"""
from __future__ import annotations

from typing import Iterator

import boto3
from botocore.config import Config
from loguru import logger

from api.config import get_settings


def make_s3_client():
    cfg = get_settings()
    if not cfg.s3_endpoint or not cfg.s3_access_key:
        return None  # S3 not configured yet
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint,
        aws_access_key_id=cfg.s3_access_key,
        aws_secret_access_key=cfg.s3_secret_key,
        region_name=cfg.s3_region or "auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def list_prefix(prefix: str, bucket: str | None = None) -> Iterator[str]:
    """Yield S3 keys under `prefix` from the configured bucket."""
    cfg = get_settings()
    s3 = make_s3_client()
    if s3 is None:
        raise RuntimeError("S3 not configured: set HETZNER_S3_* in .env")
    bucket = bucket or cfg.s3_bucket
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            yield obj["Key"]


def ping() -> bool:
    """Return True iff we can connect and list at least one bucket."""
    s3 = make_s3_client()
    if s3 is None:
        return False
    try:
        s3.list_buckets()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"S3 ping failed: {e}")
        return False
