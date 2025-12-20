#!/usr/bin/env python3
"""One-off helper to rebuild Neo4j recommendation flags and relationships.

The script loads a previously generated recommendations export (JSON plus optional
CSV), replays the `has_recommendation` attribute on `Visitor_this_year` nodes and
recreates the `IS_RECOMMENDED` relationships that connect visitors to
`Sessions_this_year`.

Designed for situations where the recommendation pipeline (step 10) cannot be
rerun but the exported outputs are still available.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

from utils.config_utils import load_config
from utils.neo4j_utils import resolve_neo4j_credentials


LOGGER = logging.getLogger("recommendation_restorer")


@dataclass
class SessionEntry:
    """Minimal slice of recommendation data needed to recreate relationships."""

    session_id: str
    similarity: float


@dataclass
class VisitorRecord:
    """Recommendation payload for a visitor."""

    badge_id: str
    sessions: List[SessionEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    visitor_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_recommendation(self) -> bool:
        return bool(self.sessions)


@dataclass
class RecommendationDataset:
    """Container for parsed export data."""

    records: List[VisitorRecord]
    generated_at: str
    show_name: str
    source_json: Path
    source_csv: Optional[Path]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, size)
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def load_recommendations_from_json(
    json_path: Path,
) -> Tuple[Dict[str, VisitorRecord], str, str]:
    if not json_path.exists():
        raise FileNotFoundError(f"Recommendations JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    recommendations = payload.get("recommendations", {}) or {}
    visitor_map: Dict[str, VisitorRecord] = {}

    for visitor_id, visitor_payload in recommendations.items():
        visitor_info = visitor_payload.get("visitor", {}) or {}
        badge_id = str(visitor_info.get("BadgeId") or visitor_id or "").strip()
        if not badge_id:
            LOGGER.warning("Skipping visitor entry with no BadgeId (key=%s)", visitor_id)
            continue

        filtered_sessions = []
        for session in visitor_payload.get("filtered_recommendations", []) or []:
            session_id = str(session.get("session_id") or session.get("sessionId") or "").strip()
            if not session_id:
                continue
            filtered_sessions.append(
                SessionEntry(
                    session_id=session_id,
                    similarity=safe_float(session.get("similarity"), 0.0),
                )
            )

        visitor_map[badge_id] = VisitorRecord(
            badge_id=badge_id,
            sessions=filtered_sessions,
            metadata=visitor_payload.get("metadata", {}) or {},
            visitor_payload=visitor_info,
        )

    dataset_metadata = payload.get("metadata", {}) or {}
    generated_at = dataset_metadata.get("generated_at") or datetime.utcnow().isoformat()
    show_name = dataset_metadata.get("show") or "unknown"

    LOGGER.info(
        "Loaded %d visitor payloads from %s (show=%s)",
        len(visitor_map),
        json_path,
        show_name,
    )

    return visitor_map, generated_at, show_name


def load_sessions_from_csv(csv_path: Path) -> Dict[str, List[SessionEntry]]:
    if not csv_path:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Recommendations CSV not found: {csv_path}")

    visitor_sessions: Dict[str, List[SessionEntry]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            visitor_id = str(row.get("visitor_id") or row.get("BadgeId") or "").strip()
            session_id = str(row.get("session_id") or "").strip()
            if not visitor_id or not session_id:
                continue
            visitor_sessions[visitor_id].append(
                SessionEntry(
                    session_id=session_id,
                    similarity=safe_float(row.get("similarity_score"), 0.0),
                )
            )

    LOGGER.info(
        "Loaded %d visitors worth of rows from CSV %s",
        len(visitor_sessions),
        csv_path,
    )
    return visitor_sessions


def build_dataset(json_path: Path, csv_path: Optional[Path]) -> RecommendationDataset:
    visitor_map, generated_at, show_name = load_recommendations_from_json(json_path)
    csv_sessions = load_sessions_from_csv(csv_path) if csv_path else {}

    for visitor_id, sessions in csv_sessions.items():
        record = visitor_map.get(visitor_id)
        if not record:
            LOGGER.warning(
                "Visitor %s found in CSV but not JSON; using CSV-only data", visitor_id
            )
            visitor_map[visitor_id] = VisitorRecord(badge_id=visitor_id, sessions=sessions)
            continue

        if not record.sessions:
            record.sessions = sessions
            continue

        json_ids = {entry.session_id for entry in record.sessions}
        csv_ids = {entry.session_id for entry in sessions}
        if json_ids != csv_ids:
            LOGGER.warning(
                "Mismatch between JSON and CSV sessions for %s (json=%d, csv=%d)",
                visitor_id,
                len(json_ids),
                len(csv_ids),
            )

    records = sorted(visitor_map.values(), key=lambda item: item.badge_id)
    return RecommendationDataset(
        records=records,
        generated_at=generated_at,
        show_name=show_name,
        source_json=json_path,
        source_csv=csv_path,
    )


class RecommendationRestorer:
    """Handles Neo4j updates for restored recommendations."""

    def __init__(self, config: Dict[str, Any], driver: Driver):
        self.config = config
        self.driver = driver
        neo_cfg = config.get("neo4j", {}) or {}
        labels = neo_cfg.get("node_labels", {}) or {}
        unique_ids = neo_cfg.get("unique_identifiers", {}) or {}

        self.visitor_label = labels.get("visitor_this_year", "Visitor_this_year")
        self.session_label = labels.get("session_this_year", "Sessions_this_year")
        self.visitor_key = unique_ids.get("visitor", "BadgeId")
        self.session_key = unique_ids.get("session", "session_id")
        self.relationship_type = "IS_RECOMMENDED"
        recommendation_cfg = config.get("recommendation", {}) or {}
        control_cfg = recommendation_cfg.get("control_group", {}) or {}
        self.control_property = control_cfg.get("neo4j_property", "control_group")
        self.default_control_value = 0
        self.show_name = (neo_cfg.get("show_name") or config.get("event", {}).get("name") or "").strip() or "unknown"

    def restore(
        self,
        dataset: RecommendationDataset,
        batch_size: int = 100,
        dry_run: bool = False,
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        records = dataset.records
        LOGGER.info(
            "Preparing to restore recommendations for %d visitors (dry_run=%s)",
            len(records),
            dry_run,
        )

        if not records:
            return {"updated_visitors": 0, "relationships_created": 0}

        if dry_run:
            with_recs = sum(1 for record in records if record.has_recommendation)
            LOGGER.info(
                "Dry-run summary: %d visitors have recommendations, %d do not",
                with_recs,
                len(records) - with_recs,
            )
            return {
                "updated_visitors": len(records),
                "relationships_created": sum(len(record.sessions) for record in records),
            }

        batch_stats = {"updated_visitors": 0, "relationships_created": 0}

        with self.driver.session() as session:
            missing_visitors = self._get_missing_ids(
                session,
                label=self.visitor_label,
                key=self.visitor_key,
                values=[record.badge_id for record in records],
            )
            if missing_visitors:
                LOGGER.warning(
                    "%d visitors from the export are missing in Neo4j and will be skipped: %s",
                    len(missing_visitors),
                    ", ".join(sorted(list(missing_visitors))[:5]) + ("..." if len(missing_visitors) > 5 else ""),
                )
                records = [record for record in records if record.badge_id not in missing_visitors]

            all_session_ids = sorted({entry.session_id for record in records for entry in record.sessions})
            missing_sessions = self._get_missing_ids(
                session,
                label=self.session_label,
                key=self.session_key,
                values=all_session_ids,
            )
            if missing_sessions:
                LOGGER.warning(
                    "%d session nodes referenced in the export are missing in Neo4j",
                    len(missing_sessions),
                )

            for batch in chunked(records, batch_size):
                batch_rows = [
                    {
                        "badge_id": record.badge_id,
                        "generated_at": dataset.generated_at,
                        "sessions": [
                            {"session_id": entry.session_id, "similarity": entry.similarity}
                            for entry in record.sessions
                            if entry.session_id not in missing_sessions
                        ],
                        "has_recommendation": record.has_recommendation,
                        "control_group": self.default_control_value,
                    }
                    for record in batch
                ]

                if replace_existing:
                    self._delete_existing_relationships(session, [row["badge_id"] for row in batch_rows])

                updated = self._update_visitors(session, batch_rows)
                created = self._create_relationships(session, batch_rows)
                batch_stats["updated_visitors"] += updated
                batch_stats["relationships_created"] += created

        LOGGER.info(
            "Finished restoring recommendations: %d visitors updated, %d relationships created",
            batch_stats["updated_visitors"],
            batch_stats["relationships_created"],
        )
        return batch_stats

    def _get_missing_ids(
        self,
        session,
        *,
        label: str,
        key: str,
        values: Sequence[str],
    ) -> set[str]:
        values = [val for val in values if val]
        if not values:
            return set()

        query = f"""
            MATCH (n:{label})
            WHERE n.{key} IN $values
            RETURN n.{key} AS value
        """
        existing = session.run(query, values=values)
        present = {record["value"] for record in existing if record["value"]}
        return set(values) - present

    def _delete_existing_relationships(self, session, badge_ids: Sequence[str]) -> None:
        if not badge_ids:
            return
        query = f"""
            MATCH (v:{self.visitor_label})
            WHERE v.{self.visitor_key} IN $badge_ids AND (v.show = $show OR v.show IS NULL)
            MATCH (v)-[rel:{self.relationship_type}]->(:{self.session_label})
            DELETE rel
        """
        session.run(query, badge_ids=badge_ids, show=self.show_name)

    def _update_visitors(self, session, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        query = f"""
            UNWIND $rows AS row
            MATCH (v:{self.visitor_label} {{{self.visitor_key}: row.badge_id}})
            WHERE v.show = $show OR v.show IS NULL
            SET v.has_recommendation = CASE WHEN row.has_recommendation THEN '1' ELSE '0' END,
                v.show = $show,
                v.recommendations_generated_at = row.generated_at,
                v.{self.control_property} = row.control_group
            RETURN count(v) AS updated
        """
        result = session.run(query, rows=rows, show=self.show_name)
        record = result.single()
        return int(record["updated"]) if record else 0

    def _create_relationships(self, session, rows: List[Dict[str, Any]]) -> int:
        rows_with_sessions = [row for row in rows if row.get("sessions")]
        if not rows_with_sessions:
            return 0
        query = f"""
            UNWIND $rows AS row
            MATCH (v:{self.visitor_label} {{{self.visitor_key}: row.badge_id}})
            WHERE v.show = $show OR v.show IS NULL
            UNWIND row.sessions AS session_data
            MATCH (s:{self.session_label} {{{self.session_key}: session_data.session_id}})
            MERGE (v)-[rel:{self.relationship_type}]->(s)
            SET rel.similarity_score = session_data.similarity,
                rel.generated_at = row.generated_at,
                rel.show = $show
            RETURN count(rel) AS created
        """
        result = session.run(query, rows=rows_with_sessions, show=self.show_name)
        record = result.single()
        return int(record["created"]) if record else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Neo4j recommendation data from exported files"
    )
    parser.add_argument(
        "--config",
        default="config/config_cpcn.yaml",
        help="Path to the YAML configuration file (defaults to CPCN config)",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the JSON export generated by session_recommendation_processor",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV export path for cross-checking recommendation rows",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of visitors to update per Neo4j batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report counts without touching Neo4j",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not delete existing IS_RECOMMENDED relationships before recreating",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_config(str(config_path))

    env_path_value = config.get("env_file", "keys/.env")
    env_file_candidates: List[Path] = []
    if env_path_value:
        env_candidate = Path(env_path_value)
        if env_candidate.is_absolute():
            env_file_candidates.append(env_candidate)
        else:
            env_file_candidates.append(config_path.parent / env_candidate)
            env_file_candidates.append(Path.cwd() / env_candidate)

    loaded_env = False
    for candidate in env_file_candidates:
        if candidate.exists():
            load_dotenv(candidate)
            LOGGER.info("Loaded environment variables from %s", candidate)
            loaded_env = True
            break
    if env_file_candidates and not loaded_env:
        LOGGER.warning(
            "Env file not found in candidates: %s",
            ", ".join(str(c) for c in env_file_candidates),
        )

    credentials = resolve_neo4j_credentials(config, logger=LOGGER)
    driver = GraphDatabase.driver(
        credentials["uri"], auth=(credentials["username"], credentials["password"])
    )

    json_path = Path(args.json)
    csv_path = Path(args.csv) if args.csv else None
    dataset = build_dataset(json_path, csv_path)

    restorer = RecommendationRestorer(config, driver)
    if dataset.show_name and dataset.show_name.lower() != restorer.show_name.lower():
        LOGGER.warning(
            "Show mismatch between export (%s) and config (%s); using config value",
            dataset.show_name,
            restorer.show_name,
        )

    try:
        stats = restorer.restore(
            dataset,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            replace_existing=not args.no_replace,
        )
        LOGGER.info("Stats: %s", stats)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
