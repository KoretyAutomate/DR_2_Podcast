"""
Unit tests for wwc_database.py.
"""

import csv
import os
import tempfile

import pytest

from dr2_podcast.research.wwc_database import WWCDatabase, WWCRating


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample WWC-format CSV file."""
    csv_path = tmp_path / "wwc_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Intervention Name", "Study Citation", "WWC Rating",
            "Improvement Index", "Domain", "Outcome Domain",
            "Study Design", "Sample Size", "Grade Level",
            "Effectiveness Rating"
        ])
        writer.writerow([
            "Reading Recovery", "Smith et al. (2019)", "Meets WWC Standards Without Reservations",
            "0.45", "Literacy", "Reading Achievement",
            "RCT", "500", "Grade 1",
            "Positive"
        ])
        writer.writerow([
            "Reading Recovery", "Johnson (2020)", "Meets WWC Standards With Reservations",
            "0.32", "Literacy", "Reading Fluency",
            "QED", "300", "Grade 1-2",
            "Potentially Positive"
        ])
        writer.writerow([
            "Success for All", "Brown & Davis (2018)", "Meets WWC Standards Without Reservations",
            "0.28", "Literacy", "Reading Achievement",
            "RCT", "1200", "Grade K-2",
            "Positive"
        ])
        writer.writerow([
            "Cognitive Tutor", "Lee (2021)", "Does Not Meet WWC Standards",
            "", "Math", "Math Achievement",
            "QED", "200", "Grade 9-12",
            "No Discernible Effects"
        ])
    return str(csv_path)


@pytest.fixture
def db(tmp_path):
    """Create an empty WWC database."""
    db_path = str(tmp_path / "test_wwc.db")
    return WWCDatabase(db_path=db_path)


@pytest.fixture
def populated_db(db, sample_csv):
    """Create a WWC database populated with sample data."""
    db.import_csv(sample_csv)
    return db


class TestWWCDatabase:

    def test_empty_db_not_populated(self, db):
        assert db.is_populated() is False
        assert db.count() == 0

    def test_import_csv(self, db, sample_csv):
        count = db.import_csv(sample_csv)
        assert count == 4
        assert db.is_populated() is True
        assert db.count() == 4

    def test_import_csv_clear_existing(self, populated_db, sample_csv):
        """Re-importing with clear_existing=True should reset."""
        count = populated_db.import_csv(sample_csv, clear_existing=True)
        assert count == 4
        assert populated_db.count() == 4

    def test_import_csv_append(self, populated_db, sample_csv):
        """Re-importing with clear_existing=False should append."""
        count = populated_db.import_csv(sample_csv, clear_existing=False)
        assert count == 4
        assert populated_db.count() == 8

    def test_import_csv_not_found(self, db):
        with pytest.raises(FileNotFoundError):
            db.import_csv("/nonexistent/path.csv")

    def test_lookup_intervention_exact(self, populated_db):
        results = populated_db.lookup_intervention("Reading Recovery")
        assert len(results) == 2
        assert all(r.intervention_name == "Reading Recovery" for r in results)

    def test_lookup_intervention_partial(self, populated_db):
        results = populated_db.lookup_intervention("Reading")
        assert len(results) == 2  # Only "Reading Recovery" matches

    def test_lookup_intervention_case_insensitive(self, populated_db):
        results = populated_db.lookup_intervention("reading recovery")
        assert len(results) == 2

    def test_lookup_intervention_no_match(self, populated_db):
        results = populated_db.lookup_intervention("Nonexistent Program")
        assert len(results) == 0

    def test_lookup_intervention_returns_ratings(self, populated_db):
        results = populated_db.lookup_intervention("Reading Recovery")
        assert isinstance(results[0], WWCRating)
        # Find the RCT one
        rct_result = [r for r in results if r.study_design == "RCT"][0]
        assert rct_result.improvement_index == 0.45
        assert rct_result.domain == "Literacy"
        assert rct_result.sample_size == 500
        assert rct_result.effectiveness_rating == "Positive"

    def test_lookup_study_by_title(self, populated_db):
        results = populated_db.lookup_study(title="Smith")
        assert len(results) == 1
        assert results[0].intervention_name == "Reading Recovery"

    def test_lookup_study_by_author(self, populated_db):
        results = populated_db.lookup_study(title="", author="Brown")
        assert len(results) == 1
        assert results[0].intervention_name == "Success for All"

    def test_lookup_study_combined(self, populated_db):
        results = populated_db.lookup_study(title="2019", author="Smith")
        assert len(results) == 1

    def test_lookup_study_no_match(self, populated_db):
        results = populated_db.lookup_study(title="Nonexistent")
        assert len(results) == 0

    def test_lookup_study_empty_params(self, populated_db):
        results = populated_db.lookup_study(title="", author="")
        assert len(results) == 0

    def test_wwcrating_dataclass(self):
        r = WWCRating(
            intervention_name="Test",
            study_citation="Author (2024)",
            wwc_rating="Meets Standards",
            improvement_index=0.5,
            domain="Math",
            study_design="RCT",
            sample_size=100,
        )
        assert r.intervention_name == "Test"
        assert r.improvement_index == 0.5

    def test_improvement_index_none_for_missing(self, populated_db):
        """Cognitive Tutor has no improvement index in the CSV."""
        results = populated_db.lookup_intervention("Cognitive Tutor")
        assert len(results) == 1
        assert results[0].improvement_index is None

    def test_close(self, db):
        db.close()
        # After closing, operations should fail
        with pytest.raises(Exception):
            db.count()


class TestCSVColumnMapping:
    """Test flexible column name recognition."""

    def test_alternative_column_names(self, tmp_path):
        """Test with different but valid column names."""
        csv_path = tmp_path / "alt_cols.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Program Name", "Citation", "Rating", "Effect Size"])
            writer.writerow(["Test Program", "Author (2024)", "Meets Standards", "0.3"])

        db_path = str(tmp_path / "test_alt.db")
        db = WWCDatabase(db_path=db_path)
        count = db.import_csv(str(csv_path))
        assert count == 1
        results = db.lookup_intervention("Test Program")
        assert len(results) == 1
        assert results[0].improvement_index == 0.3
        db.close()
