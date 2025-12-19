"""Utilities for wiring Application Insights logging across PA pipelines."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
except Exception:  # pragma: no cover - optional dependency
    configure_azure_monitor = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.resources import Resource
except Exception:  # pragma: no cover - optional dependency
    Resource = None  # type: ignore[assignment]

_APP_INSIGHTS_SENTINEL = "_PA_APP_INSIGHTS_CONFIGURED"
_RESOURCE_CREATE_PATCHED = False
_LOG_FACTORY_PATCHED = False


def _resolve_connection_string() -> Optional[str]:
    for key in (
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "AZURE_APPINSIGHTS_CONNECTION_STRING",
        "APPINSIGHTS_CONNECTION_STRING",
    ):
        value = os.getenv(key)
        if value:
            return value
    return None


def ensure_resource_factory(log: Optional[logging.Logger] = None) -> None:
    """Patch OpenTelemetry Resource factory when detectors are unavailable."""

    global _RESOURCE_CREATE_PATCHED

    if Resource is None or _RESOURCE_CREATE_PATCHED:
        return

    log = log or logging.getLogger("pa.app_insights")

    try:
        Resource.create({})
    except StopIteration:
        original_create = Resource.create
        log.warning(
            "OpenTelemetry resource detectors unavailable; falling back to minimal resource creation"
        )

        def _safe_create(
            attributes: Optional[Mapping[str, Any]] = None,
            schema_url: Optional[str] = None,
        ) -> Resource:
            try:
                return original_create(attributes=attributes, schema_url=schema_url)
            except StopIteration:
                attr_map = dict(attributes) if attributes else {}
                attr_map.setdefault("telemetry.sdk.language", "python")
                attr_map.setdefault("telemetry.sdk.name", "opentelemetry")
                attr_map.setdefault("telemetry.sdk.version", "unknown")
                service_name_env = os.getenv("OTEL_SERVICE_NAME")
                if service_name_env:
                    attr_map.setdefault("service.name", service_name_env)
                return Resource(attr_map, schema_url)

        Resource.create = staticmethod(_safe_create)  # type: ignore[assignment]

    _RESOURCE_CREATE_PATCHED = True


def ensure_source_context_in_logs(
    log: Optional[logging.Logger] = None,
) -> None:
    """Augment Python log records with file/method/line metadata."""

    global _LOG_FACTORY_PATCHED

    if _LOG_FACTORY_PATCHED:
        return

    baseline_factory = logging.getLogRecordFactory()
    emitter = log or logging.getLogger("pa.app_insights")

    def _factory(*args: Any, **kwargs: Any) -> logging.LogRecord:  # type: ignore[name-defined]
        record = baseline_factory(*args, **kwargs)
        location = f"{record.pathname}:{record.funcName}:{record.lineno}"
        record.ai_source_location = location
        record.ai_source_file = record.pathname
        record.ai_source_function = record.funcName
        record.ai_source_line = record.lineno
        if not getattr(record, "_pa_ai_prefixed", False):
            record.msg = f"[{location}] {record.msg}"
            record._pa_ai_prefixed = True
        return record

    logging.setLogRecordFactory(_factory)
    _LOG_FACTORY_PATCHED = True
    emitter.debug("Log record factory patched to include source context for Application Insights")


def configure_app_insights(
    service_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Configure Azure Monitor OpenTelemetry exporter if credentials are available.

    Args:
        service_name: Optional logical name for the emitting service. This value is
            attached to telemetry resources so entries are easy to filter in
            Application Insights.
        logger: Optional logger to emit diagnostic messages. Falls back to a
            module-level logger.

    Returns:
        bool: ``True`` when Azure Monitor integration is active, ``False`` when
        telemetry is skipped (missing package or connection string).
    """

    log = logger or logging.getLogger("pa.app_insights")
    ensure_source_context_in_logs(log)

    # If we've already configured telemetry, optionally refresh service name and exit
    if os.environ.get(_APP_INSIGHTS_SENTINEL):
        if service_name and not os.getenv("OTEL_SERVICE_NAME"):
            os.environ["OTEL_SERVICE_NAME"] = service_name
        return True

    connection_string = _resolve_connection_string()
    if not connection_string:
        log.debug("Application Insights connection string not provided; skipping telemetry setup")
        return False

    if configure_azure_monitor is None:
        log.warning("azure-monitor-opentelemetry package not installed; unable to emit telemetry")
        return False

    if Resource is None:
        log.warning("opentelemetry.sdk.resources.Resource unavailable; unable to emit telemetry")
        return False

    os.environ.setdefault("OTEL_EXPERIMENTAL_RESOURCE_DETECTORS", "otel")

    ensure_resource_factory(log)

    kwargs = {}
    resource_attributes = {"service.name": service_name} if service_name else {}
    if service_name:
        os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    kwargs["resource"] = Resource(attributes=resource_attributes)

    try:
        configure_azure_monitor(connection_string=connection_string, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Failed to configure Azure Monitor telemetry: %s", exc, exc_info=True)
        return False

    os.environ[_APP_INSIGHTS_SENTINEL] = service_name or "configured"
    log.info("Azure Monitor telemetry configured for service '%s'", service_name or "pa")
    return True


__all__ = ["configure_app_insights", "ensure_resource_factory"]
