import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Result information from staging Step 1 outputs for downstream steps."""

    event_root_dir: str
    output_dir: str
    alt_output_dir: str
    copied_files: List[str]
    missing_expected: List[str]
    support_status: Dict[str, bool]


def _index_directory(directory: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    try:
        for entry in os.listdir(directory):
            candidate = os.path.join(directory, entry)
            if os.path.isfile(candidate):
                index[entry.lower()] = candidate
    except FileNotFoundError:
        return {}
    return index


def stage_step1_outputs(
    config: Dict[str, Any],
    input_paths: Dict[str, Optional[str]],
    data_dir: str,
    logger: Any,
    *,
    support_targets: Optional[Dict[str, str]] = None,
    expected_files: Optional[Dict[str, List[str]]] = None,
) -> StageResult:
    """Copy Step 1 payloads into the local data directory for downstream steps."""

    event_name = config.get('event', {}).get('name', 'ecomm')
    event_root_dir = os.path.join(data_dir, event_name)
    output_dir = os.path.join(event_root_dir, 'output')
    alt_output_dir = os.path.join(data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(alt_output_dir, exist_ok=True)
    os.makedirs(event_root_dir, exist_ok=True)

    support_targets = support_targets or {}
    support_status = {key: False for key in support_targets.keys()}
    copied_files: List[str] = []

    filtered_inputs = {
        key: os.fspath(path)
        for key, path in input_paths.items()
        if path and os.path.isdir(os.fspath(path))
    }

    for key, original in input_paths.items():
        if not original:
            continue
        path_str = os.fspath(original)
        if not os.path.exists(path_str):
            logger.warning("%s path does not exist: %s", key, path_str)
        elif not os.path.isdir(path_str):
            logger.warning("%s path is not a directory: %s", key, path_str)
        else:
            files = os.listdir(path_str)
            logger.info("%s contains %s files", key, len(files))

    for label, path_str in filtered_inputs.items():
        for filename in os.listdir(path_str):
            source_path = os.path.join(path_str, filename)
            if not os.path.isfile(source_path):
                continue

            destination = os.path.join(output_dir, filename)
            shutil.copy2(source_path, destination)
            alt_destination = os.path.join(alt_output_dir, filename)
            shutil.copy2(source_path, alt_destination)
            copied_files.append(filename)
            logger.info("Copied %s from %s", filename, label)

            support_key = filename.lower()
            if support_key in support_targets:
                relative_target = support_targets[support_key]
                support_path = os.path.join(event_root_dir, relative_target)
                os.makedirs(os.path.dirname(support_path), exist_ok=True)
                shutil.copy2(source_path, support_path)
                support_status[support_key] = True
                logger.info("Copied support artifact %s to %s", filename, support_path)

    output_index = _index_directory(output_dir)
    alt_index = _index_directory(alt_output_dir)

    def _ensure_present(alias_list: List[str]) -> bool:
        for alias in alias_list:
            lookup = alias.lower()
            if lookup in output_index:
                return True
            if lookup in alt_index:
                src = alt_index[lookup]
                dst = os.path.join(output_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                output_index[lookup] = dst
                logger.info("Recovered %s from alternate directory", os.path.basename(src))
                return True
        return False

    missing_expected: List[str] = []
    if expected_files:
        for canonical_name, alias_list in expected_files.items():
            if not _ensure_present(alias_list):
                missing_expected.append(canonical_name)
                logger.warning("Missing expected file: %s", canonical_name)

    missing_support = [name for name, ok in support_status.items() if not ok]
    if missing_support:
        logger.warning(
            "Missing Neo4j support files from Step 1 payload: %s",
            ', '.join(sorted(missing_support)),
        )

    if not copied_files:
        logger.warning(
            "No files were copied from Step 1 outputs; downstream processors may fail"
        )
    else:
        logger.info("Staged %s files from Step 1 outputs", len(copied_files))

    return StageResult(
        event_root_dir=event_root_dir,
        output_dir=output_dir,
        alt_output_dir=alt_output_dir,
        copied_files=copied_files,
        missing_expected=missing_expected,
        support_status=support_status,
    )
