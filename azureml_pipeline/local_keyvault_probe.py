"""Utility script to mimic an Azure ML step locally.

The script loads the same environment variables that pipeline steps rely on,
connects to the configured Azure Key Vault, enumerates every secret the current
identity can access, and emits a telemetry heartbeat to Application Insights.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from logging import getLogger, INFO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PA_DIR = PROJECT_ROOT / "PA"

for path in (PROJECT_ROOT, PA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from PA.utils.app_insights import configure_app_insights, ensure_resource_factory
from PA.utils.keyvault_utils import KeyVaultManager
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import get_tracer_provider

LOG = logging.getLogger("pa.local_keyvault_probe")
logger = getLogger(__name__)
logger.setLevel(INFO)
tracer = trace.get_tracer(__name__, tracer_provider=get_tracer_provider())

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local .env file, authenticate against Azure Key Vault using "
            "the same credential chain as the pipeline, list all accessible "
            "secrets, and send a health log to Application Insights."
        )
    )
    parser.add_argument(
        "--env-file",
        default="notebooks/.env",
        help="Path to the .env file that carries the Azure ML pipeline credentials",
    )
    parser.add_argument(
        "--keyvault-name",
        default=None,
        help="Optional override for the Key Vault name (defaults to KEYVAULT_NAME)",
    )
    parser.add_argument(
        "--service-name",
        default="pa-local-keyvault-probe",
        help="Logical service name to stamp on Application Insights telemetry",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print raw secret values to stdout (never logged). Use with caution.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to dump the discovered secrets as JSON",
    )
    return parser.parse_args()


def _setup_direct_app_insights(service_name: str) -> None:
    connection_string = (
        os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        or os.getenv("AZURE_APPINSIGHTS_CONNECTION_STRING")
        or os.getenv("APPINSIGHTS_CONNECTION_STRING")
    )

    if not connection_string:
        LOG.warning("Application Insights connection string not found; skipping direct telemetry wiring")
        return

    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    os.environ.setdefault("OTEL_EXPERIMENTAL_RESOURCE_DETECTORS", "otel")

    ensure_resource_factory(LOG)

    try:
        resource = Resource(attributes={"service.name": service_name})
        configure_azure_monitor(connection_string=connection_string, resource=resource)
    except Exception as exc:  # pragma: no cover - defensive logging for local runs
        LOG.error("Failed to configure Azure Monitor telemetry: %s", exc, exc_info=True)
        return

    logger.info("Direct Azure Monitor telemetry configured for service '%s'", service_name)


def _load_env_file(env_file: str) -> None:
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Cannot locate env file at {env_file}")

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _mask_secret(value: Optional[str], reveal: bool) -> str:
    if value is None:
        return "<missing>"
    if reveal:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _collect_secrets(manager: KeyVaultManager, reveal: bool) -> List[Dict[str, Optional[str]]]:
    secrets: List[Dict[str, Optional[str]]] = []
    for props in manager.client.list_properties_of_secrets():
        secret_value = manager.get_secret(props.name)
        secrets.append(
            {
                "name": props.name,
                "enabled": props.enabled,
                "content_type": props.content_type,
                "updated_on": props.updated_on.isoformat() if props.updated_on else None,
                "value": _mask_secret(secret_value, reveal=reveal),
                "value_length": len(secret_value) if secret_value else 0,
            }
        )
    return secrets


def _print_table(secrets: List[Dict[str, Optional[str]]]) -> None:
    if not secrets:
        print("No secrets returned by Key Vault.")
        return

    header = f"{'Name':<40} {'Enabled':<8} {'Updated (UTC)':<25} {'Length':<6} Value"
    print(header)
    print("-" * len(header))
    for entry in secrets:
        print(
            f"{entry['name']:<40} "
            f"{str(entry['enabled']):<8} "
            f"{(entry['updated_on'] or '-'): <25} "
            f"{entry['value_length']:<6} "
            f"{entry['value']}"
        )


def _write_json_dump(secrets: List[Dict[str, Optional[str]]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(secrets, indent=2))
    LOG.info("Wrote JSON dump with %s entries to %s", len(secrets), path)
    logger.info("Wrote JSON dump with %s entries to %s", len(secrets), path)

def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _load_env_file(args.env_file)
    keyvault_name = args.keyvault_name or os.getenv("KEYVAULT_NAME")
    if not keyvault_name:
        raise RuntimeError("KEYVAULT_NAME is not set in the environment")

    LOG.info("Loaded environment from %s", args.env_file)
    logger.info("Loaded environment from %s", args.env_file)

    _setup_direct_app_insights(args.service_name)

    configure_app_insights(service_name=args.service_name, logger=LOG)

    manager = KeyVaultManager(keyvault_name)
    secrets = _collect_secrets(manager, reveal=args.show_values)
    _print_table(secrets)

    if args.json_output:
        _write_json_dump(secrets, args.json_output)

    LOG.info("Enumerated %s secrets from Key Vault '%s'", len(secrets), keyvault_name)
    logger.info("Enumerated %s secrets from Key Vault '%s'", len(secrets), keyvault_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
