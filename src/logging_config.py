"""Consistent terminal logging for app and CLI entry points."""
from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    logger = logging.getLogger("src")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if any(getattr(handler, "_gita_handler", False) for handler in logger.handlers):
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[source=%(request_source)s client=%(client_id)s request=%(request_id)s]: %(message)s",
        datefmt="%H:%M:%S",
    ))

    class RequestContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            from src.request_context import current_request_context

            context = current_request_context()
            record.request_source = context["source"]
            record.client_id = context["client_id"]
            record.request_id = context["request_id"]
            return True

    handler.addFilter(RequestContextFilter())
    handler._gita_handler = True
    logger.addHandler(handler)
