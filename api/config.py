"""Centralized configuration for the orchestrator API.

All values come from environment variables (loaded from .env in dev).
Pydantic Settings validates types and surfaces missing-required-var errors
on startup instead of at first use.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── API auth ───
    api_key: str = Field(..., alias="API_KEY")

    # ─── Database ───
    postgres_host: str = Field("postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("fotoklass", alias="POSTGRES_DB")
    postgres_user: str = Field("fotoklass", alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")

    # ─── RunPod ───
    runpod_api_key: str = Field(..., alias="RUNPOD_API_KEY")
    runpod_rtx4090_per_hour: float = Field(0.34, alias="RUNPOD_RTX4090_PER_HOUR")

    # ─── Hetzner Object Storage (S3 API) ───
    s3_endpoint: str = Field("", alias="HETZNER_S3_ENDPOINT")
    s3_access_key: str = Field("", alias="HETZNER_S3_ACCESS_KEY")
    s3_secret_key: str = Field("", alias="HETZNER_S3_SECRET_KEY")
    s3_region: str = Field("", alias="HETZNER_S3_REGION")
    s3_bucket: str = Field("", alias="HETZNER_S3_BUCKET")

    # ─── Cost tariffs ───
    hetzner_storage_per_gb: float = Field(0.01, alias="HETZNER_OBJECT_STORAGE_PER_GB")
    hetzner_egress_per_tb: float = Field(1.00, alias="HETZNER_EGRESS_PER_TB")
    vps_per_month: float = Field(10.0, alias="VPS_PER_MONTH")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
