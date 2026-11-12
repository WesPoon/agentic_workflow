"""DynamoDBStore that accepts the plain Python values our tools write.

boto3's DynamoDB serializer rejects float (and NaN/inf have no DynamoDB representation), and
returns numbers as Decimal. This wrapper converts float → Decimal (NaN/inf → None) on put and
Decimal → int/float on read, so tool code and prompts see ordinary numbers.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from langgraph.store.base import Item, Op, PutOp, Result
from langgraph_checkpoint_aws import DynamoDBStore


def to_dynamo(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return Decimal(repr(value)) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_dynamo(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dynamo(v) for v in value]
    return value


def from_dynamo(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: from_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_dynamo(v) for v in value]
    return value


def _fix_result(res: Result) -> Result:
    if isinstance(res, Item):
        res.value = from_dynamo(res.value)
    elif isinstance(res, list):
        for item in res:
            if isinstance(item, Item):
                item.value = from_dynamo(item.value)
    return res


class SafeDynamoDBStore(DynamoDBStore):
    @staticmethod
    def _convert(ops):
        return [op._replace(value=to_dynamo(op.value)) if isinstance(op, PutOp) and op.value is not None else op
                for op in ops]

    def batch(self, ops: list[Op]) -> list[Result]:
        return [_fix_result(r) for r in super().batch(self._convert(list(ops)))]

    async def abatch(self, ops: list[Op]) -> list[Result]:
        return [_fix_result(r) for r in await super().abatch(self._convert(list(ops)))]
