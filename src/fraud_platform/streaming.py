from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def serialize_event(event: BaseModel) -> bytes:
    return event.model_dump_json().encode("utf-8")


def deserialize_event(payload: bytes, model: type[T]) -> T:
    return model.model_validate_json(payload.decode("utf-8"))
