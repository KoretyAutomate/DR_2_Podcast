"""
What Works Clearinghouse (WWC) local database for education intervention quality ratings.

WWC is maintained by the Institute of Education Sciences (IES) and provides
systematic reviews of education research. This module stores WWC ratings
in a local SQLite database for offline lookup during the social science pipeline.

Usage:
    1. Download data from https://ies.ed.gov/ncee/wwc/StudyFindings (export CSV)
    2. Import: WWCDatabase().import_csv("path/to/wwc.csv")
    3. Lookup: WWCDatabase().lookup_intervention("Reading Recovery")
"""

import csv
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.expanduser("~/.cache/dr2podcast/wwc.db")


@dataclass
class WWCRating:
    """A single WWC study finding/rating."""
    intervention_name: str
    study_citation: str
    wwc_rating: str           # "Meets WWC Standards Without Reservations", etc.
    improvement_index: Optional[float] = None
    domain: str = ""          # e.g., "Literacy", "Math", "Science"
    outcome_domain: str = ""
    study_design: str = ""    # "RCT", "QED", etc.
    sample_size: Optional[int] = None
    grade_level: str = ""
    effectiveness_rating: str = ""  # "Positive", "Potentially Positive", etc.


class WWCDatabase:
    """Local SQLite database for WWC education intervention quality ratings."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS wwc_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_name TEXT NOT NULL,
                study_citation TEXT NOT NULL,
                wwc_rating TEXT NOT NULL,
                improvement_index REAL,
                domain TEXT,
                outcome_domain TEXT,
                study_design TEXT,
                sample_size INTEGER,
                grade_level TEXT,
                effectiveness_rating TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wwc_intervention
            ON wwc_ratings (intervention_name COLLATE NOCASE)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wwc_citation
            ON wwc_ratings (study_citation COLLATE NOCASE)
        """)
        self.conn.commit()

    def import_csv(self, csv_path: str, clear_existing: bool = True) -> int:
        """Import WWC data from a CSV file.

        Returns the number of records imported.
        The CSV should have columns matching the WWC export format.
        Column mapping is flexible — common WWC column names are recognized.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if clear_existing:
            self.conn.execute("DELETE FROM wwc_ratings")

        count = 0
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize header names (strip whitespace, lowercase)
            if reader.fieldnames:
                col_map = {c.strip().lower(): c for c in reader.fieldnames}
            else:
                logger.error("CSV has no headers")
                return 0

            for row in reader:
                # Flexible column mapping
                intervention = self._get_col(row, col_map, [
                    "intervention name", "intervention", "program name", "program"
                ])
                citation = self._get_col(row, col_map, [
                    "study citation", "citation", "study", "study reference"
                ])
                rating = self._get_col(row, col_map, [
                    "wwc rating", "rating", "study rating", "wwc study rating"
                ])

                if not intervention or not rating:
                    continue

                imp_idx = self._safe_float(self._get_col(row, col_map, [
                    "improvement index", "effect size", "improvement"
                ]))
                sample = self._safe_int(self._get_col(row, col_map, [
                    "sample size", "n", "total sample"
                ]))

                self.conn.execute(
                    "INSERT INTO wwc_ratings "
                    "(intervention_name, study_citation, wwc_rating, improvement_index, "
                    "domain, outcome_domain, study_design, sample_size, grade_level, "
                    "effectiveness_rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        intervention,
                        citation or "",
                        rating,
                        imp_idx,
                        self._get_col(row, col_map, ["domain", "outcome area"]) or "",
                        self._get_col(row, col_map, ["outcome domain", "outcome"]) or "",
                        self._get_col(row, col_map, ["study design", "design"]) or "",
                        sample,
                        self._get_col(row, col_map, ["grade level", "grade", "education level"]) or "",
                        self._get_col(row, col_map, ["effectiveness rating", "effectiveness"]) or "",
                    ),
                )
                count += 1

        self.conn.commit()
        logger.info(f"WWC database: imported {count} records from {csv_path.name}")
        return count

    def lookup_intervention(self, name: str) -> List[WWCRating]:
        """Fuzzy match against intervention names.

        Returns all matching ratings, ordered by intervention name similarity.
        Uses LIKE with wildcards for partial matching.
        """
        rows = self.conn.execute(
            "SELECT intervention_name, study_citation, wwc_rating, improvement_index, "
            "domain, outcome_domain, study_design, sample_size, grade_level, "
            "effectiveness_rating FROM wwc_ratings "
            "WHERE intervention_name LIKE ? COLLATE NOCASE "
            "ORDER BY intervention_name",
            (f"%{name}%",),
        ).fetchall()
        return [self._row_to_rating(r) for r in rows]

    def lookup_study(self, title: str, author: str = "") -> List[WWCRating]:
        """Search by study title and/or author in the citation field."""
        query_parts = []
        params = []
        if title:
            query_parts.append("study_citation LIKE ? COLLATE NOCASE")
            params.append(f"%{title}%")
        if author:
            query_parts.append("study_citation LIKE ? COLLATE NOCASE")
            params.append(f"%{author}%")

        if not query_parts:
            return []

        where_clause = " AND ".join(query_parts)
        rows = self.conn.execute(
            f"SELECT intervention_name, study_citation, wwc_rating, improvement_index, "
            f"domain, outcome_domain, study_design, sample_size, grade_level, "
            f"effectiveness_rating FROM wwc_ratings WHERE {where_clause}",
            params,
        ).fetchall()
        return [self._row_to_rating(r) for r in rows]

    def is_populated(self) -> bool:
        """Check if data has been imported."""
        row = self.conn.execute("SELECT COUNT(*) FROM wwc_ratings").fetchone()
        return row[0] > 0

    def count(self) -> int:
        """Return total number of records."""
        row = self.conn.execute("SELECT COUNT(*) FROM wwc_ratings").fetchone()
        return row[0]

    def close(self):
        self.conn.close()

    @staticmethod
    def _row_to_rating(row: tuple) -> WWCRating:
        return WWCRating(
            intervention_name=row[0],
            study_citation=row[1],
            wwc_rating=row[2],
            improvement_index=row[3],
            domain=row[4] or "",
            outcome_domain=row[5] or "",
            study_design=row[6] or "",
            sample_size=row[7],
            grade_level=row[8] or "",
            effectiveness_rating=row[9] or "",
        )

    @staticmethod
    def _get_col(row: dict, col_map: dict, candidates: list) -> str:
        """Get a column value by trying multiple candidate names."""
        for name in candidates:
            orig = col_map.get(name.lower())
            if orig and row.get(orig):
                return row[orig].strip()
        return ""

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        if not val:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(val: str) -> Optional[int]:
        if not val:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None
