"""Validate model-generated calls without exposing rejected arguments for execution."""

import json
import math
from typing import Any


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def parse_tool_arguments(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return either executable arguments or a fixed, non-sensitive error code."""
    if arguments is None:
        return None, "missing_arguments"
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (ValueError, TypeError, RecursionError):
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "non_object_arguments"
    try:
        if not _is_json_value(parsed):
            return None, "non_json_values"
    except RecursionError:
        return None, "non_json_values"
    return parsed, None


def rejected_tool_call(name: Any, call_id: Any, code: str) -> dict[str, str]:
    """Bound model-controlled identifiers before using them in diagnostics."""

    def identifier(value: Any) -> str:
        if not isinstance(value, str) or not 0 < len(value) <= 128:
            return ""
        if not all(
            char.isascii() and (char.isalnum() or char in "_.-") for char in value
        ):
            return ""
        return value

    return {"name": identifier(name), "call_id": identifier(call_id), "code": code}
