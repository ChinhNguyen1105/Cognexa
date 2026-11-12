from __future__ import annotations

import math
from typing import Any


def require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def require_non_negative_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def require_finite_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def require_unit_interval(value: Any, field_name: str) -> None:
    require_finite_number(value, field_name)

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")