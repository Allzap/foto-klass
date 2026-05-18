"""API-key auth via Authorization: Bearer <key> header."""
from fastapi import Header, HTTPException, status

from api.config import get_settings


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    cfg = get_settings()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. "
                   "Expected: 'Authorization: Bearer <API_KEY>'",
        )
    token = authorization[len("Bearer "):].strip()
    if token != cfg.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
