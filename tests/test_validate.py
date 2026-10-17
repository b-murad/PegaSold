"""Tests for the data validator (src/validate.py) on the saved processed arrays."""
import json
from pathlib import Path

import pytest

from src import validate

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(not (DATA / "preprocessing_report.json").exists(),
                                reason="run `python -m src.data_prep` first")


def test_processed_data_passes_every_check():
    report, names, arrays, tiers, tables = validate.load(DATA)
    v = validate.validate(report, names, arrays, tiers, tables)
    assert v.failed == 0 and v.passed > 50


def test_validator_flags_a_wrong_split_size():
    report, names, arrays, tiers, tables = validate.load(DATA)
    report["split_rows"]["train"] += 1
    v = validate.validate(report, names, arrays, tiers, tables)
    assert v.failed >= 1
