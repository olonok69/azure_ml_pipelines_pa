#!/usr/bin/env python
"""Utilities to adapt CPCN/CPC datasets to the legacy GraphQL schema used by the pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("transform_cpc_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

RUN_TIMESTAMP = datetime.utcnow().replace(microsecond=0).isoformat()
RUN_ID = str(uuid.uuid4()).upper()
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://closerstillmedia.com/schema-adapter")
PROVIDER_SOURCE = "schema_adapter"
DEFAULT_RECORD_END_DATE = "9999-12-31T00:00:00"

SHOW_DETAILS: Dict[str, Dict[str, Optional[str]]] = {
    "CPCN24": {
        "event_id": "72d62f37-e0c0-44b5-ad3a-b1a1009e9842",
        "show_date": None,
    },
    "CPCN25": {
        "event_id": "aaddecf2-1634-4d4d-b704-b2ad009a7c54",
        "show_date": "2025-11-21T00:00:00",
    },
    "CPC2025": {
        "event_id": "aa071bfa-1209-48e0-b62b-b23500e07d05",
        "show_date": "2025-05-09T00:00:00",
    },
}

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "cpcn"
REFERENCE_DIR = DATA_DIR / "reference"

REFERENCE_DEMOGRAPHICS = REFERENCE_DIR / "30092025_CPC25_demographics_graphql.json"
REFERENCE_REGISTRATION = REFERENCE_DIR / "30092025_CPCN25_registration_graphql.json"

ID_SUFFIX_DELIMITER = "|"


def select_input_file(*candidates: str) -> Path:
    """Return the first existing file from the provided candidate list."""

    for candidate in candidates:
        path = DATA_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate files exist: {', '.join(str(DATA_DIR / c) for c in candidates)}"
    )


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
        buffer = []
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
                    elif char in "[\n\r\t ":
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


def apply_metadata(record: Dict[str, Any], show_ref: Optional[str], seq_key: str, seq_value: int, source_filename: str) -> None:
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
        details = SHOW_DETAILS.get(show_ref or "")
        if details:
            record["metadata_show_date"] = details.get("show_date")


def resequence(records: Sequence[Dict[str, Any]], seq_key: str) -> None:
    for idx, record in enumerate(records, start=1):
        record[seq_key] = idx


def make_demographic_id(badge_id: Optional[str], question_id: Optional[int], answer_id: Optional[int], show_ref: Optional[str]) -> str:
    parts = [badge_id or "", str(question_id or ""), str(answer_id or ""), show_ref or ""]
    return str(uuid.uuid5(ID_NAMESPACE, ID_SUFFIX_DELIMITER.join(parts)))


def transform_flat_demographics(records: Sequence[Dict[str, Any]], template_keys: Sequence[str], show_ref: str, source_filename: str) -> List[Dict[str, Any]]:
    template = build_template(template_keys)
    event_id = SHOW_DETAILS.get(show_ref, {}).get("event_id")
    transformed: List[Dict[str, Any]] = []
    for idx, item in enumerate(records, start=1):
        record = template.copy()
        badge_id = item.get("badge_id") or item.get("badgeId")
        question_id = safe_int(item.get("question_id"))
        answer_id = safe_int(item.get("answer_id"))
        record["id"] = make_demographic_id(badge_id, question_id, answer_id, show_ref)
        record["badge_id"] = badge_id
        record["event_id"] = item.get("event_id") or event_id
        record["show_ref"] = (item.get("show_ref") or show_ref or "UNKNOWN").upper()
        record["question_type"] = item.get("question_type")
        record["question_id"] = question_id
        record["question_text"] = item.get("question_text")
        record["answer_id"] = answer_id
        record["answer_text"] = item.get("answer_text")
        record["is_positive"] = bool_to_flag(item.get("is_positive"))
        record["response_text"] = item.get("answer_specification_text") or item.get("response_text")
        record.setdefault("metadata_demographics_dimension_checksum", 0)
        apply_metadata(record, record["show_ref"], "demographics_seq", idx, source_filename)
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

        record: Dict[str, Any] = {
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
        legacy_records.append(record)
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


def flatten_question_responses(payload: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    responses = payload.get("questionResponses") or []
    for resp in responses:
        yield payload, resp


def transform_nested_demographics(records: Sequence[Dict[str, Any]], template_keys: Sequence[str], show_ref: str, source_filename: str) -> List[Dict[str, Any]]:
    template = build_template(template_keys)
    event_id = SHOW_DETAILS.get(show_ref, {}).get("event_id")
    transformed: List[Dict[str, Any]] = []
    counter = 0
    for base in records:
        badge_id = base.get("badgeId")
        inferred_show = infer_show_ref(base.get("source"), show_ref)
        for payload, response in flatten_question_responses(base):
            counter += 1
            record = template.copy()
            question_id = safe_int(response.get("questionId"))
            answer_id = safe_int(response.get("answerId"))
            record["id"] = make_demographic_id(badge_id, question_id, answer_id, inferred_show)
            record["badge_id"] = badge_id
            record["event_id"] = payload.get("eventId") or event_id
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


def transform_registration(records: Sequence[Dict[str, Any]], template_keys: Sequence[str], show_ref: str, source_filename: str) -> List[Dict[str, Any]]:
    template = build_template(template_keys)
    event_id = SHOW_DETAILS.get(show_ref, {}).get("event_id")
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


def ensure_registration_processor_compatibility(records: Sequence[Dict[str, Any]], required_keys: Sequence[str], dataset_name: str) -> None:
    sample = records[0] if records else {}
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(f"{dataset_name}: missing required keys for registration processor {missing}")


def main(include_legacy: bool = False) -> None:
    LOGGER.info("Run id %s triggered at %s", RUN_ID, RUN_TIMESTAMP)
    ref_demo_keys = load_reference_keys(REFERENCE_DEMOGRAPHICS)
    ref_reg_keys = load_reference_keys(REFERENCE_REGISTRATION)
    if "show_ref" not in ref_reg_keys:
        ref_reg_keys.append("show_ref")

    output_files = {
        "cpcn_demographics": DATA_DIR / "20251201_CPCN24_25_demographics_graphql.json",
        "cpcn_registration": DATA_DIR / "20251201_eventregistration_CPCN24_25_graphql.json",
        "cpc_demographics": DATA_DIR / "20251201_eventregistration_CPC25_demographics_graphql.json",
        "cpc_registration": DATA_DIR / "20251201_eventregistration_CPC25_graphql.json",
    }

    legacy_output_files = {
        "cpcn_demographics": DATA_DIR / "20251201_CPCN24_25_demographics_legacy.json",
        "cpcn_registration": DATA_DIR / "20251201_eventregistration_CPCN24_25_legacy.json",
        "cpc_demographics": DATA_DIR / "20251201_eventregistration_CPC25_demographics_legacy.json",
        "cpc_registration": DATA_DIR / "20251201_eventregistration_CPC25_legacy.json",
    }

    LOGGER.info("Processing CPCN demographic datasets")
    cpcn24_demo_path = DATA_DIR / "20241007_CPCN24_demographics.json"
    cpcn25_demo_path = select_input_file(
        "20251201_eventregistration_CPCN25_demographics.json",
    )

    cpcn24_demo = load_json_array(cpcn24_demo_path)
    cpcn25_demo_raw = load_json_array(cpcn25_demo_path)
    demo_records: List[Dict[str, Any]] = []
    demo_records.extend(
        transform_flat_demographics(cpcn24_demo, ref_demo_keys, "CPCN24", cpcn24_demo_path.name)
    )

    if cpcn25_demo_raw and isinstance(cpcn25_demo_raw[0], dict) and "questionResponses" in cpcn25_demo_raw[0]:
        demo_records.extend(
            transform_nested_demographics(cpcn25_demo_raw, ref_demo_keys, "CPCN25", cpcn25_demo_path.name)
        )
    else:
        demo_records.extend(
            transform_flat_demographics(cpcn25_demo_raw, ref_demo_keys, "CPCN25", cpcn25_demo_path.name)
        )
    resequence(demo_records, "demographics_seq")
    validate_against_reference(demo_records, ref_demo_keys, "CPCN demographics")
    ensure_registration_processor_compatibility(
        demo_records,
        ["badge_id", "question_text", "answer_text", "show_ref", "metadata_source_filename"],
        "CPCN demographics",
    )
    save_json_array(output_files["cpcn_demographics"], demo_records)
    LOGGER.info("Saved %s records to %s", len(demo_records), output_files["cpcn_demographics"].name)

    LOGGER.info("Processing CPCN registration datasets")
    cpcn24_reg_path = DATA_DIR / "20251006_eventregistration_CPCN24.json"
    cpcn25_reg_path = select_input_file(

        "20251201_eventregistration_CPCN25.json",
    )

    cpcn24_reg = load_json_array(cpcn24_reg_path)
    cpcn25_reg = load_json_array(cpcn25_reg_path)
    reg_records: List[Dict[str, Any]] = []
    reg_records.extend(
        transform_registration(cpcn24_reg, ref_reg_keys, "CPCN24", cpcn24_reg_path.name)
    )
    reg_records.extend(
        transform_registration(cpcn25_reg, ref_reg_keys, "CPCN25", cpcn25_reg_path.name)
    )
    resequence(reg_records, "registration_seq")
    validate_against_reference(
        reg_records,
        [key for key in ref_reg_keys if key != "show_ref"],
        "CPCN registration",
        allow_extra=["show_ref"],
    )
    ensure_registration_processor_compatibility(
        reg_records,
        ["id", "forename", "surname", "email", "company", "job_title", "badge_type", "badge_id", "reg_code", "source", "attended", "show_ref"],
        "CPCN registration",
    )
    save_json_array(output_files["cpcn_registration"], reg_records)
    LOGGER.info("Saved %s records to %s", len(reg_records), output_files["cpcn_registration"].name)

    LOGGER.info("Processing CPC demographics dataset")
    cpc_demo_nested = load_json_array(DATA_DIR / "20251006_eventregistration_CPC25_demographics.json")
    cpc_demo_records = transform_nested_demographics(cpc_demo_nested, ref_demo_keys, "CPC2025", "20251006_eventregistration_CPC25_demographics.json")
    resequence(cpc_demo_records, "demographics_seq")
    validate_against_reference(cpc_demo_records, ref_demo_keys, "CPC demographics")
    ensure_registration_processor_compatibility(
        cpc_demo_records,
        ["badge_id", "question_text", "answer_text", "show_ref", "metadata_source_filename"],
        "CPC demographics",
    )
    save_json_array(output_files["cpc_demographics"], cpc_demo_records)
    LOGGER.info("Saved %s records to %s", len(cpc_demo_records), output_files["cpc_demographics"].name)

    LOGGER.info("Processing CPC registration dataset")
    cpc_reg = load_json_array(DATA_DIR / "20251006_eventregistration_CPC25.json")
    cpc_reg_records = transform_registration(cpc_reg, ref_reg_keys, "CPC2025", "20251006_eventregistration_CPC25.json")
    resequence(cpc_reg_records, "registration_seq")
    validate_against_reference(
        cpc_reg_records,
        [key for key in ref_reg_keys if key != "show_ref"],
        "CPC registration",
        allow_extra=["show_ref"],
    )
    ensure_registration_processor_compatibility(
        cpc_reg_records,
        ["id", "forename", "surname", "email", "company", "job_title", "badge_type", "badge_id", "reg_code", "source", "attended", "show_ref"],
        "CPC registration",
    )
    save_json_array(output_files["cpc_registration"], cpc_reg_records)
    LOGGER.info("Saved %s records to %s", len(cpc_reg_records), output_files["cpc_registration"].name)

    if include_legacy:
        LOGGER.info("Generating legacy-format datasets")
        cpcn_demo_legacy = convert_demographics_to_legacy(demo_records)
        save_json_array(legacy_output_files["cpcn_demographics"], cpcn_demo_legacy)
        LOGGER.info("Saved %s records to %s", len(cpcn_demo_legacy), legacy_output_files["cpcn_demographics"].name)

        cpcn_reg_legacy = convert_registration_to_legacy(reg_records)
        save_json_array(legacy_output_files["cpcn_registration"], cpcn_reg_legacy)
        LOGGER.info("Saved %s records to %s", len(cpcn_reg_legacy), legacy_output_files["cpcn_registration"].name)

        cpc_demo_legacy = convert_demographics_to_legacy(cpc_demo_records)
        save_json_array(legacy_output_files["cpc_demographics"], cpc_demo_legacy)
        LOGGER.info("Saved %s records to %s", len(cpc_demo_legacy), legacy_output_files["cpc_demographics"].name)

        cpc_reg_legacy = convert_registration_to_legacy(cpc_reg_records)
        save_json_array(legacy_output_files["cpc_registration"], cpc_reg_legacy)
        LOGGER.info("Saved %s records to %s", len(cpc_reg_legacy), legacy_output_files["cpc_registration"].name)

    LOGGER.info("Transformation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize CPCN/CPC registration datasets")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also emit datasets using the legacy schema expected when old_format=true",
    )

    args = parser.parse_args()
    main(include_legacy=args.legacy)
