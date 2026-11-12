"""Fetch secrets from AWS Secrets Manager into the process environment.

Each pair (ENV_NAME, ENV_NAME_SECRET_ARN) is checked once at startup: if ENV_NAME is unset and the
ARN variable is set, the secret value is loaded. A JSON secret with exactly one key is unwrapped.
Nothing is logged except the names.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

SECRET_ENV_NAMES = ("ANTHROPIC_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY", "POLYGON_API_KEY", "GITHUB_TOKEN")


def load_secrets_into_env(region_name: str | None = None) -> list[str]:
    loaded: list[str] = []
    pending = [n for n in SECRET_ENV_NAMES if not os.environ.get(n) and os.environ.get(f"{n}_SECRET_ARN")]
    if not pending:
        return loaded
    import boto3  # imported lazily so local runs need no boto3

    client = boto3.client("secretsmanager", region_name=region_name)
    for name in pending:
        arn = os.environ[f"{name}_SECRET_ARN"]
        resp = client.get_secret_value(SecretId=arn)
        value = resp.get("SecretString") or ""
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict) and len(parsed) == 1:
                value = str(next(iter(parsed.values())))
        except (json.JSONDecodeError, TypeError):
            pass
        os.environ[name] = value
        loaded.append(name)
        log.info("loaded secret %s from Secrets Manager", name)
    return loaded
