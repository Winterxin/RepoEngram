"""Deterministic conversion of domain dataclasses to JSON-compatible values."""

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in fields(value)
            if not (field.metadata.get("omit_empty") and getattr(value, field.name) == "")
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    return value
