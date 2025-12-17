"""Utilities for wiring Application Insights logging across PA pipelines."""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
except Exception:  # pragma: no cover - optional dependency
    configure_azure_monitor = None  # type: ignore[assignment]

_APP_INSIGHTS_SENTINEL = "_PA_APP_INSIGHTS_CONFIGURED"


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

    kwargs = {}
    if service_name:
        kwargs["resource"] = {"service.name": service_name}
        os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    try:
        configure_azure_monitor(connection_string=connection_string, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Failed to configure Azure Monitor telemetry: %s", exc, exc_info=True)
        return False

    os.environ[_APP_INSIGHTS_SENTINEL] = service_name or "configured"
    log.info("Azure Monitor telemetry configured for service '%s'", service_name or "pa")
    return True


__all__ = ["configure_app_insights"]
