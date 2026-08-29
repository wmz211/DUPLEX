"""Read API credentials from environment variables without exposing values."""
from __future__ import annotations

import os


API_KEY_ENV_VARS = {
    "bocha": "BOCHA_API_KEY",
    "qwen": "QWEN_API_KEY",
}


def require_api_keys(*services: str) -> dict[str, str]:
    """Return requested API keys or raise an error listing missing variables."""
    unknown = [service for service in services if service not in API_KEY_ENV_VARS]
    if unknown:
        raise ValueError(f"Unknown API service(s): {', '.join(sorted(unknown))}")

    values = {
        service: os.environ.get(API_KEY_ENV_VARS[service], "").strip()
        for service in services
    }
    missing = [API_KEY_ENV_VARS[service] for service, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )
    return values
