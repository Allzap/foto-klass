"""Thin wrapper over the RunPod REST API.

Two operations are needed for MVP:
  - whoami / ping        — to surface auth errors at startup
  - list_gpu_types       — discover available 4090s
Later (auto-launch phase) we add:
  - create_pod, get_pod_status, terminate_pod
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from api.config import get_settings

GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"
REST_ENDPOINT = "https://rest.runpod.io/v1"


@dataclass
class RunPodAccount:
    user_id: str
    email: str
    current_spend_per_hr: float


class RunPodClient:
    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or get_settings().runpod_api_key
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                GRAPHQL_ENDPOINT,
                headers=self._headers,
                json={"query": query, "variables": variables or {}},
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
            return data["data"]

    def whoami(self) -> RunPodAccount:
        data = self._graphql("query { myself { id email currentSpendPerHr } }")
        m = data["myself"]
        return RunPodAccount(
            user_id=m["id"],
            email=m["email"],
            current_spend_per_hr=float(m.get("currentSpendPerHr", 0) or 0),
        )

    def ping(self) -> bool:
        try:
            self.whoami()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RunPod ping failed: {e}")
            return False
