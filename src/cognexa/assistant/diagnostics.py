from __future__ import annotations

import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any


LOGGER_NAME = "cognexa.diagnostics"


def configure_diagnostics() -> logging.Logger:
    """Enable opt-in local diagnostic logging when COGNEXA_DIAGNOSTICS=1."""
    logger = logging.getLogger(LOGGER_NAME)
    if os.getenv("COGNEXA_DIAGNOSTICS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not any(getattr(handler, "_cognexa_diagnostics", False) for handler in logger.handlers):
        Path("logs").mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler("logs/diagnostics.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        handler._cognexa_diagnostics = True
        logger.addHandler(handler)

    return logger


def _safe_repr(value: Any, limit: int = 12000) -> str:
    rendered = pformat(value, width=110, compact=False)
    if len(rendered) > limit:
        return rendered[:limit] + f"\n... [truncated; {len(rendered)} chars total]"
    return rendered


def _log_result(logger: logging.Logger, stage: str, result: Any) -> None:
    logger.debug("[%s] RESULT\n%s", stage, _safe_repr(result))


def instrument_method(component: Any, method_name: str, stage: str) -> None:
    """Wrap one already-created component method without changing its return value."""
    if component is None or not hasattr(component, method_name):
        return

    logger = logging.getLogger(LOGGER_NAME)
    if logger.disabled:
        return

    original = getattr(component, method_name)
    marker = f"_cognexa_diagnostic_wrapped_{method_name}"
    if getattr(component, marker, False):
        return

    def wrapped(*args, **kwargs):
        logger.debug("[%s] INPUT args=%s kwargs=%s",
                     stage, _safe_repr(args), _safe_repr(kwargs))
        result = original(*args, **kwargs)
        _log_result(logger, stage, result)
        return result

    setattr(component, method_name, wrapped)
    setattr(component, marker, True)


def attach_diagnostics(
    *,
    adaptive_retriever=None,
    answer_extractor=None,
    answer_selector=None,
    answer_verifier=None,
    answerability_checker=None,
) -> None:
    """Instrument the live Cognexa pipeline for observation only."""
    logger = configure_diagnostics()
    if logger.disabled:
        return

    logger.info("========== A5 DIAGNOSTICS ENABLED ==========")
    instrument_method(adaptive_retriever, "retrieve", "RETRIEVAL")
    instrument_method(answer_extractor, "extract_candidates", "EXTRACTOR")
    instrument_method(answer_verifier, "verify", "VERIFIER")
    instrument_method(answer_selector, "select_detailed", "SELECTOR")
    instrument_method(answerability_checker, "assess", "ANSWERABILITY")
