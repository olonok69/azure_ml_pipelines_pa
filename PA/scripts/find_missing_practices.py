#!/usr/bin/env python3
"""Identify registration companies absent from the practices missing catalogue.

This utility loads the registration datasets for LVA and BVA, cleans the
company names by stripping punctuation and whitespace, and compares them
against the practices missing list. Any company present in the registration
files but missing from ``practices_missing.csv`` is reported.

Usage (from the repository root)::

    python -m PA.scripts.find_missing_practices \
        --bva "PA/data/lva/csv/Registration_data_bva.csv" \
        --lva "PA/data/lva/csv/Registration_data_lva.csv" \
        --practices "PA/data/lva/practices_missing.csv" \
        --output "missing_companies.csv"

If ``--output`` is omitted, the script prints the missing company list to STDOUT.
"""

from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

import pandas as pd


@dataclass(frozen=True)
class CompanyRecord:
    """Container for the original and normalised company names."""

    original: str
    normalised: str
    source: str


def normalise_name(value: str) -> str:
    """Normalise a company name by removing punctuation/spaces and lowering case."""
    if not isinstance(value, str):
        return ""

    cleaned = "".join(ch for ch in value.lower() if ch not in string.punctuation and not ch.isspace())
    return cleaned.strip()


def load_company_records(csv_path: Path, column: str, source: str) -> Iterable[CompanyRecord]:
    """Yield company records from the provided CSV column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    for value in df[column].dropna():
        original = str(value).strip()
        normalised = normalise_name(original)
        if normalised:
            yield CompanyRecord(original=original, normalised=normalised, source=source)


def unique_by_normalised(records: Iterable[CompanyRecord]) -> dict[str, CompanyRecord]:
    """Return a mapping of normalised name -> representative record."""
    unique: dict[str, CompanyRecord] = {}
    for record in records:
        unique.setdefault(record.normalised, record)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bva",
        type=Path,
        default=Path("PA/data/lva/csv/Registration_data_bva.csv"),
        help="Path to the BVA registration CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--lva",
        type=Path,
        default=Path("PA/data/lva/csv/Registration_data_lva.csv"),
        help="Path to the LVA registration CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--practices",
        type=Path,
        default=Path("PA/data/lva/practices_missing.csv"),
        help="Path to the practices missing CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write the missing companies report",
    )
    return parser.parse_args()


def build_missing_companies(args: argparse.Namespace) -> list[dict[str, str]]:
    registration_records = unique_by_normalised(
        list(load_company_records(args.bva, "Company", "BVA"))
        + list(load_company_records(args.lva, "Company", "LVA"))
    )

    practices_records = unique_by_normalised(
        load_company_records(args.practices, "Company Name", "practices_missing")
    )
    practices_keys: Set[str] = set(practices_records.keys())

    missing = [
        {
            "Company": record.original,
            "NormalisedCompany": record.normalised,
            "Source": record.source,
        }
        for record in registration_records.values()
        if record.normalised not in practices_keys
    ]

    missing.sort(key=lambda row: row["NormalisedCompany"])
    return missing


def main() -> None:
    args = parse_args()
    missing = build_missing_companies(args)

    if args.output:
        output_df = pd.DataFrame(missing)
        output_df.to_csv(args.output, index=False)
        print(f"Wrote {len(missing)} missing companies to {args.output}")
    else:
        if not missing:
            print("All registration companies are present in practices_missing.csv")
            return

        print("Companies missing from practices_missing.csv:\n")
        for row in missing:
            print(f"- {row['Company']} (source: {row['Source']})")
        print(f"\nTotal missing companies: {len(missing)}")


if __name__ == "__main__":
    main()
