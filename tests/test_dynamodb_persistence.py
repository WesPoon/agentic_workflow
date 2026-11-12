"""PERSISTENCE=dynamodb: DynamoDBSaver + DynamoDBStore against mocked DynamoDB/S3, through a crash
and a resume on a fresh graph instance (as on a new microVM)."""
from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from tests.test_pipeline import ScriptedModel


def _create_checkpoint_table(name: str) -> None:
    boto3.client("dynamodb", region_name="us-east-1").create_table(
        TableName=name, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                              {"AttributeName": "SK", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
    )


def test_dynamodb_crash_and_resume(tmp_path, monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-east-1",
                 "AWS_REGION": "us-east-1", "PERSISTENCE": "dynamodb", "CHECKPOINT_TABLE": "ckpt",
                 "STORE_TABLE": "store", "ARTIFACT_BUCKET": "bkt", "STORAGE": "local", "COMPUTE_BACKEND": "local",
                 "WORK_DIR": str(tmp_path / "runs"), "MAX_REVISIONS": "2"}.items():
        monkeypatch.setenv(k, v)
    from src.config import get_settings
    get_settings.cache_clear()
    import src.nodes.common as common
    monkeypatch.setattr(common, "get_llm", lambda: ScriptedModel())
    ScriptedModel.calls, ScriptedModel.fail_once_on, ScriptedModel.last_by_phase = [], "review:alpha_mom:1", {}
    paper = tmp_path / "p.pdf"
    paper.write_bytes(b"%PDF-1.4")

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bkt")
        _create_checkpoint_table("ckpt")
        from langgraph_checkpoint_aws import DynamoDBStore
        DynamoDBStore(table_name="store", region_name="us-east-1").setup()

        from src import runner
        from src.graph import build_graph
        from src.persistence import make_checkpointer, make_store
        tid = "ddb_thread_0000000000000000000000001"
        with pytest.raises(RuntimeError):
            runner.start_run([str(paper)], thread_id=tid, graph=build_graph(make_checkpointer(), make_store()))
        runner.resume_run(tid, graph=build_graph(make_checkpointer(), make_store()))

        status = json.loads((tmp_path / "runs" / tid / "status.json").read_text())
        assert status["status"] == "complete" and status["approved"] == ["alpha_mom"]
        store = make_store()
        assert store.get(("runs", "manifests"), tid).value["strategies_skipped"] == ["beta_mom"]
        assert len(store.search(("review", "technique_outcomes"))) == 2
        outcome = store.search(("review", "technique_outcomes"))[0].value
        assert type(outcome["oos_sharpe"]) in (int, float)  # no Decimal leaks into prompts
    get_settings.cache_clear()
