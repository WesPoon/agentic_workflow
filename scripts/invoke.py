"""Call the deployed AgentCore runtime.

  python scripts/invoke.py run papers/a.pdf papers/b.pdf
  python scripts/invoke.py status <thread_id>
  python scripts/invoke.py resume <thread_id>

Needs AGENT_RUNTIME_ARN and AWS_REGION in the environment (or .env).
"""
from __future__ import annotations

import json
import os
import sys
import uuid

import boto3
from dotenv import load_dotenv

load_dotenv()


def call(payload: dict, session_hint: str = "") -> dict:
    client = boto3.client("bedrock-agentcore", region_name=os.environ["AWS_REGION"])
    session_id = f"momentum-{session_hint or 'x'}-{uuid.uuid4().hex}"[:100]   # must be >= 33 chars
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode(),
    )
    body = resp["response"].read() if hasattr(resp.get("response"), "read") else resp.get("response")
    return json.loads(body) if body else {}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        print(json.dumps(call({"action": "run", "paper_keys": args}), indent=2))
    elif cmd in ("status", "resume"):
        print(json.dumps(call({"action": cmd, "thread_id": args[0]}, args[0][-20:]), indent=2))
    else:
        print(__doc__)
        sys.exit(1)
