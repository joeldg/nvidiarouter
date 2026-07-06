# @spec[OBSERVABILITY.md#Requirements]
"""
Central logging configuration.

Standardises the whole application on structlog so the gateway, router, and
agent layers all emit consistent, structured key=value (or JSON) logs.
"""

import logging

import structlog


# @spec[OBSERVABILITY.md#Requirements]
def configure_logging(level: str = "info", json_logs: bool = False) -> None:
    """Configure structlog (and the stdlib root logger) once for the process."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", level=log_level)

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Configure with sensible defaults on import; callers may reconfigure.
try:  # pragma: no cover - defensive import-time setup
    from .config import settings

    configure_logging(settings.log_level, settings.log_json)
except Exception:  # noqa: BLE001
    configure_logging()
