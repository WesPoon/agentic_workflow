"""Create the LangGraph Store table once: python scripts/setup_store.py [--table momentum-store]."""
import argparse
import os

from langgraph_checkpoint_aws import DynamoDBStore

p = argparse.ArgumentParser()
p.add_argument("--table", default=os.environ.get("STORE_TABLE", "momentum-store"))
p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
a = p.parse_args()
DynamoDBStore(table_name=a.table, region_name=a.region).setup()
print(f"store table {a.table} ready in {a.region}")
