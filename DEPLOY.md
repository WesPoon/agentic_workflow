# Deploying the momentum pipeline to AWS

The agent graph runs on **Amazon Bedrock AgentCore Runtime**. Each backtest attempt is committed to a
GitHub branch and run by a **self-hosted GitHub Actions runner on your EC2**. State lives in
**DynamoDB**, files in **S3**.

```
you ──invoke──▶ AgentCore Runtime (main.py, LangGraph)
                 ├─ Bedrock (Claude)                 LLM calls
                 ├─ DynamoDB  momentum-checkpoints   graph checkpoints (resume after crash)
                 ├─ DynamoDB  momentum-store         cross-run memory
                 ├─ S3        runs/<thread_id>/...   briefs, strategies, reviews, status.json
                 └─ GitHub    run/<thread>/<strategy>-r<rev>-<job>  ──push──▶ Actions on your EC2
                                                                               └─ S3 result.json, metrics, tearsheet
```

## Run locally first

```bash
cp .env.example .env            # defaults to Bedrock Opus 5.5: needs AWS credentials, no API key
uv pip install -r requirements.txt
python -m pytest tests -q
python -m src.runner run --papers papers/<one>.pdf
python -m src.runner status <thread_id>
python -m src.runner resume <thread_id>     # after a crash
python -m src.runner inspect                # cross-run memory
```

## One-time AWS setup

Pick one region that has AgentCore and the Claude model. Enable the model in the Bedrock console,
then confirm the inference profile id:

```bash
aws bedrock list-inference-profiles --query "inferenceProfileSummaries[?contains(inferenceProfileId,'claude-opus-5-5')].inferenceProfileId"
```

Deploy the infrastructure stack and create the Store table:

```bash
aws cloudformation deploy --stack-name momentum \
  --template-file infra/cloudformation/momentum-infra.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides BucketName=<unique-bucket-name>
aws cloudformation describe-stacks --stack-name momentum --query "Stacks[0].Outputs"
python scripts/setup_store.py --region <region>
```

Put secret values in (the stack creates them with a placeholder):

```bash
aws secretsmanager put-secret-value --secret-id momentum/github-token --secret-string '<fine-grained PAT>'
aws secretsmanager put-secret-value --secret-id momentum/polygon-api-key --secret-string '<key>'   # optional
```

Upload papers:

```bash
aws s3 cp papers/ s3://<bucket>/papers/ --recursive --exclude "*" --include "*.pdf"
```

## GitHub + EC2 backtest runner

1. Push this project to a **private** repo with `main` as the default branch. `main` must not contain a
   `jobs/` or `strategies/` folder at the repo root.
2. Create a fine-grained PAT: this repo only, **Contents: read and write**. Store it as above.
3. Attach the instance profile `momentum-backtest-runner` (stack output) to your EC2. Its security group
   needs **no inbound rules**; it needs outbound HTTPS.
4. Provision it:

```bash
cp infra/ansible/inventory.ini.example infra/ansible/inventory.ini   # fill in host and key
TOKEN=$(gh api -X POST repos/<owner>/<repo>/actions/runners/registration-token --jq .token)
ansible-playbook -i infra/ansible/inventory.ini infra/ansible/backtest_runner.yml \
  -e github_repo=<owner>/<repo> -e runner_token=$TOKEN -e aws_region=<region> \
  -e polygon_secret_arn=<PolygonKeySecretArn>     # optional
```

5. Smoke test the runner path from your Mac (uses the S3 bucket and GitHub, no LLM):

```bash
STORAGE=s3 ARTIFACT_BUCKET=<bucket> COMPUTE_BACKEND=github GITHUB_REPO=<owner>/<repo> \
GITHUB_TOKEN=<pat> AWS_REGION=<region> python - <<'PY'
from pathlib import Path
from src.jobs.runner import run_job
from src.workdir import run_paths
p = run_paths("smoke").ensure()
f = p.strategies / "synthetic.py"; f.write_text(Path("tests/jobs/synthetic_strategy.py").read_text())
print(run_job({"thread_id": "smoke", "strategy_name": "synthetic", "revision": 0, "attempt": 1,
               "params": {"seed": 1}, "change_note": "runner smoke test"}, f, p)[:400])
PY
```

A branch `run/smoke/synthetic-r0-<id>` appears, the workflow runs on the `backtest` runner, and the
result prints within a minute or two.

## Deploy the agent

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -n momentum_pipeline -e main.py -rf requirements-agent.txt \
  --region <region> --disable-memory --idle-timeout 900 --max-lifetime 28800
```

Attach the managed policy `momentum-agent-runtime` (stack output `AgentRuntimePolicyArn`) to the
execution role that `configure` created or printed. Then test the container locally and deploy:

```bash
ENV="--env LLM_PROVIDER=bedrock --env LLM_MODEL_ID=us.anthropic.claude-opus-5-5 --env LLM_EFFORT=high \
     --env PERSISTENCE=dynamodb --env STORAGE=s3 --env ARTIFACT_BUCKET=<bucket> \
     --env COMPUTE_BACKEND=github --env GITHUB_REPO=<owner>/<repo> \
     --env GITHUB_TOKEN_SECRET_ARN=<GitHubTokenSecretArn> --env WORK_DIR=/tmp/work \
     --env AWS_REGION=<region>"
agentcore launch --local $ENV
agentcore launch $ENV          # CodeBuild → ECR → runtime; prints the runtime ARN
```

## Run it

```bash
export AWS_REGION=<region> AGENT_RUNTIME_ARN=<arn>
python scripts/invoke.py run papers/a.pdf papers/b.pdf     # returns thread_id immediately
python scripts/invoke.py status <thread_id>                # started → running(node) → complete | failed
python scripts/invoke.py resume <thread_id>                # continue a failed or reaped run
aws s3 cp s3://<bucket>/runs/<thread_id>/attempts.csv -     # every attempt: commit, change, OOS Sharpe, verdict
```

## Configuration reference

Runtime environment variables (`agentcore launch --env`):

| Variable | Cloud value | Notes |
|---|---|---|
| `AWS_REGION` | your region | required |
| `LLM_PROVIDER` | `bedrock` | no Anthropic API key needed; `anthropic` also works with `ANTHROPIC_API_KEY_SECRET_ARN` |
| `LLM_MODEL_ID` | `us.anthropic.claude-opus-5-5` | change model here and in the stack's two model parameters |
| `LLM_EFFORT` | `high` | Opus 5.5 thinking is always on; its default effort is `medium` |
| `PERSISTENCE` | `dynamodb` | |
| `CHECKPOINT_TABLE`, `STORE_TABLE` | defaults `momentum-checkpoints`, `momentum-store` | |
| `STORAGE` | `s3` | |
| `ARTIFACT_BUCKET` | stack output | also used for checkpoint offload |
| `WORK_DIR` | `/tmp/work` | microVM disk is ephemeral; S3 is the record |
| `COMPUTE_BACKEND` | `github` | |
| `GITHUB_REPO` | `owner/repo` | |
| `GITHUB_TOKEN_SECRET_ARN` | stack output | |
| `POLYGON_API_KEY_SECRET_ARN` | stack output | only if the agent itself needs Polygon; the EC2 gets it from Ansible |
| `BACKTEST_TIMEOUT_S` | `1800` | keep below the workflow's `timeout-minutes` |
| `MAX_REVISIONS` | `3` | failed reviews before a strategy is skipped |
| `MAX_PHASE_STEPS` | `80` | LLM calls per agent phase before giving up on it |
| `RECURSION_LIMIT` | `300` | whole-graph step cap |

Where each secret lives:

| Secret | Store | Read by |
|---|---|---|
| GitHub PAT | Secrets Manager `momentum/github-token` | agent runtime |
| Polygon key | Secrets Manager `momentum/polygon-api-key` | EC2 worker (and agent if configured) |
| Anthropic key | not needed with Bedrock | |
| AWS credentials | none stored: IAM roles on the runtime and the EC2 | |
| Runner registration token | one-time, passed to Ansible | |

## Limits to know

- AgentCore microVM: 2 vCPU / 8 GB, 8 h per session. A run that hits 8 h stops; `resume` continues
  from the last checkpoint in a new session.
- One backtest at a time on the EC2 (workflow concurrency group). Each attempt adds roughly 20–60 s of
  commit, pickup and S3 round-trip overhead.
- The evaluator, not the agent, computes every metric. More attempts lower the deflated Sharpe
  probability by design.
