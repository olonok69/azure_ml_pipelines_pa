#!/usr/bin/env python
"""Utilities to adapt LVS/LVA datasets to the GraphQL schema used by the vet pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("transform_lva_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

RUN_TIMESTAMP = datetime.utcnow().replace(microsecond=0).isoformat()
RUN_ID = str(uuid.uuid4()).upper()
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://closerstillmedia.com/schema-adapter")
PROVIDER_SOURCE = "schema_adapter"
DEFAULT_RECORD_END_DATE = "9999-12-31T00:00:00"

SHOW_DETAILS: Dict[str, Dict[str, Optional[str]]] = {
    "LVS24": {
        "event_id": "9d802127-ee7f-4367-bb4c-b07600eb3d8c",
        "show_date": "2024-11-20T00:00:00",
    },
    "LVS25": {
        "event_id": "2719f487-00b5-43be-b251-b1ed00ff8f5d",
        "show_date": "2025-11-20T00:00:00",
    },
}

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "lva"
# Reuse the CPC reference exports because the GraphQL schema is identical across shows.
REFERENCE_DIR = ROOT_DIR / "data" / "cpcn" / "reference"

REFERENCE_DEMOGRAPHICS = REFERENCE_DIR / "30092025_CPC25_demographics_graphql.json"
REFERENCE_REGISTRATION = REFERENCE_DIR / "30092025_CPCN25_registration_graphql.json"

ID_SUFFIX_DELIMITER = "|"


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_array(path: Path, payload: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(payload), handle, ensure_ascii=False, indent=2)


def bool_to_flag(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return "1" if bool(value) else "0"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return "1"
        if normalized in {"0", "false", "no", "n"}:
            return "0"
    return None


def safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def normalize_timestamp(raw: Any) -> Optional[str]:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt_obj = datetime.fromisoformat(text)
    except ValueError:
        return str(raw)
    return dt_obj.replace(tzinfo=None).isoformat(timespec="seconds")


def flag_to_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    str_val = str(value).strip().lower()
    if str_val in {"1", "true", "yes", "y"}:
        return "Yes"
    if str_val in {"0", "false", "no", "n"}:
        return "No"
    return str(value)


def infer_show_ref(source: Optional[str], default: Optional[str]) -> Optional[str]:
    if source:
        matches = re.findall(r"[A-Za-z]{2,}\d{2,}", source)
        if matches:
            return matches[0].upper()
    return default


def load_reference_keys(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        buffer: List[str] = []
        depth = 0
        inside_object = False
        while True:
            chunk = handle.read(1024)
            if not chunk:
                break
            for char in chunk:
                if not inside_object:
                    if char == "{":
                        inside_object = True
                        depth = 1
                        buffer.append(char)
                    elif char in "[{\n\r\t ":
                        continue
                    else:
                        continue
                else:
                    buffer.append(char)
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            obj = json.loads("".join(buffer))
                            keys = list(obj.keys())
                            LOGGER.info("Loaded %s reference keys from %s", len(keys), path.name)
                            return keys
        raise ValueError(f"Unable to locate first object in reference file {path}")


def build_template(keys: Iterable[str]) -> Dict[str, Any]:
    return {key: None for key in keys}


def apply_metadata(
    record: Dict[str, Any],
    show_ref: Optional[str],
    seq_key: str,
    seq_value: int,
    source_filename: str,
) -> None:
    record["metadata_source_filename"] = source_filename
    record["metadata_provider_source"] = PROVIDER_SOURCE
    record["metadata_pipeline_run_id"] = RUN_ID
    record["metadata_pipeline_trigger_date"] = RUN_TIMESTAMP
    record["metadata_record_start_date"] = RUN_TIMESTAMP
    record["metadata_record_end_date"] = DEFAULT_RECORD_END_DATE
    record["metadata_is_current"] = "1"
    record["metadata_is_updated"] = "0"
    if seq_key:
        record[seq_key] = seq_value
    if "metadata_show_date" in record and not record.get("metadata_show_date"):
        details = SHOW_DETAILS.get((show_ref or "").upper())
        if details:
            record["metadata_show_date"] = details.get("show_date")


def resequence(records: Sequence[Dict[str, Any]], seq_key: str) -> None:
    for idx, record in enumerate(records, start=1):
        record[seq_key] = idx


def make_demographic_id(
    badge_id: Optional[str],
    question_id: Optional[int],
    answer_id: Optional[int],
    show_ref: Optional[str],
) -> str:
    parts = [badge_id or "", str(question_id or ""), str(answer_id or ""), show_ref or ""]
    return str(uuid.uuid5(ID_NAMESPACE, ID_SUFFIX_DELIMITER.join(parts)))


def transform_nested_demographics(
    records: Sequence[Dict[str, Any]],
    template_keys: Sequence[str],
    show_ref: str,
    source_filename: str,
) -> List[Dict[str, Any]]:
    template = build_template(template_keys)
    event_id = SHOW_DETAILS.get(show_ref.upper(), {}).get("event_id")
    transformed: List[Dict[str, Any]] = []
    counter = 0

    for base in records:
        badge_id = base.get("badgeId")
        inferred_show = infer_show_ref(base.get("source"), show_ref)
        responses = base.get("questionResponses") or []
        for response in responses:
            counter += 1
            record = template.copy()
            question_id = safe_int(response.get("questionId"))
            answer_id = safe_int(response.get("answerId"))
            record["id"] = make_demographic_id(badge_id, question_id, answer_id, inferred_show)
            record["badge_id"] = badge_id
            record["event_id"] = base.get("eventId") or event_id
            record["show_ref"] = (inferred_show or show_ref or "UNKNOWN").upper()
            record["question_type"] = response.get("questionType")
            record["question_id"] = question_id
            record["question_text"] = response.get("questionText")
            record["answer_id"] = answer_id
            record["answer_text"] = response.get("answerText")
            record["is_positive"] = bool_to_flag(response.get("isPositive"))
            record["response_text"] = response.get("responseText")
            record.setdefault("metadata_demographics_dimension_checksum", 0)
            apply_metadata(record, record["show_ref"], "demographics_seq", counter, source_filename)
            transformed.append(record)
    return transformed


def transform_registration(
    records: Sequence[Dict[str, Any]],
    template_keys: Sequence[str],
    show_ref: str,
    source_filename: str,
) -> List[Dict[str, Any]]:
    template = build_template(template_keys)
    event_id = SHOW_DETAILS.get(show_ref.upper(), {}).get("event_id")
    transformed: List[Dict[str, Any]] = []

    for idx, item in enumerate(records, start=1):
        record = template.copy()
        record["id"] = item.get("id")
        record["title"] = item.get("title")
        record["forename"] = item.get("forename")
        record["surname"] = item.get("surname")
        record["email"] = item.get("email")
        record["tel"] = item.get("tel")
        record["mobile"] = item.get("mobile")
        record["fax"] = item.get("fax")
        record["company"] = item.get("company")
        record["job_title"] = item.get("jobTitle") or item.get("job_title")
        record["addr1"] = item.get("addr1")
        record["addr2"] = item.get("addr2")
        record["addr3"] = item.get("addr3")
        record["town"] = item.get("town")
        record["county"] = item.get("county")
        record["postcode"] = item.get("postcode")
        record["country"] = item.get("country")
        record["status"] = item.get("status")
        record["badge_type"] = item.get("badgeType") or item.get("badge_type")
        record["event_id"] = item.get("eventId") or event_id
        record["registration_date"] = normalize_timestamp(item.get("registrationDate"))
        record["badge_id"] = item.get("badgeId")
        record["reg_code"] = item.get("regCode")
        record["source"] = item.get("source")
        record["attended"] = bool_to_flag(item.get("attended"))
        record["last_modified_date"] = normalize_timestamp(item.get("lastModifiedDate"))
        inferred_show = infer_show_ref(record["source"], show_ref)
        record["show_ref"] = (inferred_show or show_ref or "UNKNOWN").upper()
        record.setdefault("metadata_dimension_checksum", 0)
        apply_metadata(record, record["show_ref"], "registration_seq", idx, source_filename)
        transformed.append(record)
    return transformed


def convert_registration_to_legacy(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    legacy_records: List[Dict[str, Any]] = []
    for item in records:
        def _as_str(key: str) -> Optional[str]:
            value = item.get(key)
            if value is None:
                return None
            return str(value)

        legacy_records.append(
            {
                "Id": item.get("id"),
                "Title": item.get("title"),
                "Forename": item.get("forename"),
                "Surname": item.get("surname"),
                "Email": item.get("email"),
                "Tel": _as_str("tel"),
                "Mobile": _as_str("mobile"),
                "Fax": _as_str("fax"),
                "Company": item.get("company"),
                "JobTitle": item.get("job_title"),
                "Addr1": item.get("addr1"),
                "Addr2": item.get("addr2"),
                "Addr3": item.get("addr3"),
                "Town": item.get("town"),
                "County": item.get("county"),
                "Postcode": item.get("postcode"),
                "Country": item.get("country"),
                "Status": item.get("status"),
                "BadgeType": item.get("badge_type"),
                "ShowRef": item.get("show_ref"),
                "RegistrationDate": item.get("registration_date"),
                "BadgeId": item.get("badge_id"),
                "RegCode": item.get("reg_code"),
                "Source": item.get("source"),
                "Attended": flag_to_yes_no(item.get("attended")),
                "SegmentFromReg": item.get("segment_from_reg"),
                "Upgrade": item.get("upgrade"),
                "METADATA_fileName": item.get("metadata_source_filename"),
            }
        )
    return legacy_records


def convert_demographics_to_legacy(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    legacy_records: List[Dict[str, Any]] = []
    for item in records:
        answer_text = item.get("answer_text")
        response_text = item.get("response_text")
        legacy_records.append(
            {
                "showref": (item.get("show_ref") or "").upper(),
                "BadgeId": item.get("badge_id"),
                "QuestionText": item.get("question_text") or "",
                "AnswerText": answer_text if answer_text not in (None, "") else (response_text or ""),
                "METADATA_fileName": item.get("metadata_source_filename"),
            }
        )
    return legacy_records


def validate_against_reference(
    records: Sequence[Dict[str, Any]],
    reference_keys: Sequence[str],
    dataset_name: str,
    allow_extra: Optional[Sequence[str]] = None,
) -> None:
    if not records:
        LOGGER.warning("%s: no records available for validation", dataset_name)
        return
    ref_set = set(reference_keys)
    record_keys = set(records[0].keys())
    missing = ref_set - record_keys
    if missing:
        raise ValueError(f"{dataset_name}: missing keys compared to reference: {sorted(missing)}")
    allowed = set(allow_extra or [])
    extra = record_keys - ref_set - allowed
    if extra:
        LOGGER.warning("%s: additional keys not in reference: %s", dataset_name, sorted(extra))


def ensure_registration_processor_compatibility(
    records: Sequence[Dict[str, Any]],
    required_keys: Sequence[str],
    dataset_name: str,
) -> None:
    sample = records[0] if records else {}
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(f"{dataset_name}: missing required keys for registration processor {missing}")


def _copy_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [deepcopy(item) for item in records]


def main(include_legacy: bool = False) -> None:
    LOGGER.info("Run id %s triggered at %s", RUN_ID, RUN_TIMESTAMP)
    ref_demo_keys = load_reference_keys(REFERENCE_DEMOGRAPHICS)
    ref_reg_keys = load_reference_keys(REFERENCE_REGISTRATION)
    if "show_ref" not in ref_reg_keys:
        ref_reg_keys.append("show_ref")

    input_files = {
        "lvs24_registration": DATA_DIR / "20251104_eventregistration_LVS24.json",
        "lvs25_registration": DATA_DIR / "20251127_eventregistration_LVS25.json",
        "lvs24_demographics": DATA_DIR / "20251104_eventregistration_LVS24_demographics.json",
        "lvs25_demographics": DATA_DIR / "20251127_eventregistration_LVS25_demographics.json",
    }

    output_files = {
        "lvs24_registration": DATA_DIR / "20251104_eventregistration_LVS24_graphql.json",
        "lvs25_registration": DATA_DIR / "20251127_eventregistration_LVS25_graphql.json",
        "combined_registration": DATA_DIR / "20251127_eventregistration_LVS24_25_graphql.json",
        "lvs24_demographics": DATA_DIR / "20251104_eventregistration_LVS24_demographics_graphql.json",
        "lvs25_demographics": DATA_DIR / "20251127_eventregistration_LVS25_demographics_graphql.json",
        "combined_demographics": DATA_DIR / "20251127_eventregistration_LVS24_25_demographics_graphql.json",
    }

    legacy_output_files = {
        "lvs24_registration": DATA_DIR / "20251104_eventregistration_LVS24_legacy.json",
        "lvs25_registration": DATA_DIR / "20251127_eventregistration_LVS25_legacy.json",
        "combined_registration": DATA_DIR / "20251127_eventregistration_LVS24_25_legacy.json",
        "lvs24_demographics": DATA_DIR / "20251104_eventregistration_LVS24_demographics_legacy.json",
        "lvs25_demographics": DATA_DIR / "20251127_eventregistration_LVS25_demographics_legacy.json",
        "combined_demographics": DATA_DIR / "20251127_eventregistration_LVS24_25_demographics_legacy.json",
    }

    LOGGER.info("Processing LVS24 demographics dataset")
    lvs24_demo_raw = load_json_array(input_files["lvs24_demographics"])
    lvs24_demo_records = transform_nested_demographics(
        lvs24_demo_raw,
        ref_demo_keys,
        "LVS24",
        input_files["lvs24_demographics"].name,
    )
    resequence(lvs24_demo_records, "demographics_seq")
    validate_against_reference(lvs24_demo_records, ref_demo_keys, "LVS24 demographics")
    ensure_registration_processor_compatibility(
        lvs24_demo_records,
        ["badge_id", "question_text", "answer_text", "show_ref", "metadata_source_filename"],
        "LVS24 demographics",
    )
    save_json_array(output_files["lvs24_demographics"], lvs24_demo_records)
    LOGGER.info("Saved %s records to %s", len(lvs24_demo_records), output_files["lvs24_demographics"].name)

    LOGGER.info("Processing LVS25 demographics dataset")
    lvs25_demo_raw = load_json_array(input_files["lvs25_demographics"])
    lvs25_demo_records = transform_nested_demographics(
        lvs25_demo_raw,
        ref_demo_keys,
        "LVS25",
        input_files["lvs25_demographics"].name,
    )
    resequence(lvs25_demo_records, "demographics_seq")
    validate_against_reference(lvs25_demo_records, ref_demo_keys, "LVS25 demographics")
    ensure_registration_processor_compatibility(
        lvs25_demo_records,
        ["badge_id", "question_text", "answer_text", "show_ref", "metadata_source_filename"],
        "LVS25 demographics",
    )
    save_json_array(output_files["lvs25_demographics"], lvs25_demo_records)
    LOGGER.info("Saved %s records to %s", len(lvs25_demo_records), output_files["lvs25_demographics"].name)

    LOGGER.info("Processing LVS24 registration dataset")
    lvs24_reg_raw = load_json_array(input_files["lvs24_registration"])
    lvs24_reg_records = transform_registration(
        lvs24_reg_raw,
        ref_reg_keys,
        "LVS24",
        input_files["lvs24_registration"].name,
    )
    resequence(lvs24_reg_records, "registration_seq")
    validate_against_reference(
        lvs24_reg_records,
        [key for key in ref_reg_keys if key != "show_ref"],
        "LVS24 registration",
        allow_extra=["show_ref"],
    )
    ensure_registration_processor_compatibility(
        lvs24_reg_records,
        [
            "id",
            "forename",
            "surname",
            "email",
            "company",
            "job_title",
            "badge_type",
            "badge_id",
            "reg_code",
            "source",
            "attended",
            "show_ref",
        ],
        "LVS24 registration",
    )
    save_json_array(output_files["lvs24_registration"], lvs24_reg_records)
    LOGGER.info("Saved %s records to %s", len(lvs24_reg_records), output_files["lvs24_registration"].name)

    LOGGER.info("Processing LVS25 registration dataset")
    lvs25_reg_raw = load_json_array(input_files["lvs25_registration"])
    lvs25_reg_records = transform_registration(
        lvs25_reg_raw,
        ref_reg_keys,
        "LVS25",
        input_files["lvs25_registration"].name,
    )
    resequence(lvs25_reg_records, "registration_seq")
    validate_against_reference(
        lvs25_reg_records,
        [key for key in ref_reg_keys if key != "show_ref"],
        "LVS25 registration",
        allow_extra=["show_ref"],
    )
    ensure_registration_processor_compatibility(
        lvs25_reg_records,
        [
            "id",
            "forename",
            "surname",
            "email",
            "company",
            "job_title",
            "badge_type",
            "badge_id",
            "reg_code",
            "source",
            "attended",
            "show_ref",
        ],
        "LVS25 registration",
    )
    save_json_array(output_files["lvs25_registration"], lvs25_reg_records)
    LOGGER.info("Saved %s records to %s", len(lvs25_reg_records), output_files["lvs25_registration"].name)

    LOGGER.info("Building combined demographics dataset")
    combined_demo_records = _copy_records(lvs24_demo_records) + _copy_records(lvs25_demo_records)
    resequence(combined_demo_records, "demographics_seq")
    save_json_array(output_files["combined_demographics"], combined_demo_records)
    LOGGER.info(
        "Saved %s records to %s",
        len(combined_demo_records),
        output_files["combined_demographics"].name,
    )

    LOGGER.info("Building combined registration dataset")
    combined_reg_records = _copy_records(lvs24_reg_records) + _copy_records(lvs25_reg_records)
    resequence(combined_reg_records, "registration_seq")
    save_json_array(output_files["combined_registration"], combined_reg_records)
    LOGGER.info(
        "Saved %s records to %s",
        len(combined_reg_records),
        output_files["combined_registration"].name,
    )

    if include_legacy:
        LOGGER.info("Generating legacy-format datasets")
        datasets_for_legacy = {
            "lvs24_registration": lvs24_reg_records,
            "lvs25_registration": lvs25_reg_records,
            "combined_registration": combined_reg_records,
        }
        for key, records in datasets_for_legacy.items():
            legacy_payload = convert_registration_to_legacy(records)
            save_json_array(legacy_output_files[key], legacy_payload)
            LOGGER.info(
                "Saved %s records to %s",
                len(legacy_payload),
                legacy_output_files[key].name,
            )

        demo_datasets_for_legacy = {
            "lvs24_demographics": lvs24_demo_records,
            "lvs25_demographics": lvs25_demo_records,
            "combined_demographics": combined_demo_records,
        }
        for key, records in demo_datasets_for_legacy.items():
            legacy_payload = convert_demographics_to_legacy(records)
            save_json_array(legacy_output_files[key], legacy_payload)
            LOGGER.info(
                "Saved %s records to %s",
                len(legacy_payload),
                legacy_output_files[key].name,
            )

    LOGGER.info("Transformation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize LVS registration datasets")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also emit datasets using the legacy schema expected when old_format=true",
    )
    args = parser.parse_args()
    main(include_legacy=args.legacy)
