"""Web demo (src/web/web_demo.py): feature parity with the saved matrix + request validation."""
import numpy as np
import pandas as pd
import pytest

from src.data_prep import build_features
from src.web.web_demo import METRICS, PROCESSED, EstimateService

pytestmark = pytest.mark.skipif(not (PROCESSED / "train_clean.csv").exists() or not METRICS.exists(),
                                reason="run `python -m src.run_all` first")

GOOD = {"category": "Yeni tikili", "location": "Nərimanov r.", "area_m2": 85, "rooms": 3,
        "floor": 5, "n_floors": 16, "has_repair": True, "has_bill_of_sale": True, "has_mortgage": False}


@pytest.fixture(scope="module")
def service():
    return EstimateService()


def test_rules_rebuild_the_saved_validation_matrix(service):
    valid = pd.read_csv(PROCESSED / "valid_clean.csv").head(200)
    rebuilt = build_features(valid, service.rules).to_numpy()
    np.testing.assert_allclose(rebuilt, np.load(PROCESSED / "X_valid.npy")[:200], atol=1e-9)


def test_prediction_shape_and_sanity(service):
    out = service.predict(GOOD)
    assert out["tier"] in {"Premium", "Standard"} and 0.5 <= out["tier_probability"] <= 1
    assert 10_000 < out["price_azn"] < 5_000_000
    assert out["price_per_m2_azn"] == round(out["price_azn"] / 85)


def test_bigger_flat_costs_more(service):
    small = service.predict({**GOOD, "area_m2": 40, "rooms": 1})["price_azn"]
    big = service.predict({**GOOD, "area_m2": 160, "rooms": 5})["price_azn"]
    assert big > small


@pytest.mark.parametrize("patch", [{"area_m2": 5}, {"area_m2": 5000}, {"rooms": 0}, {"rooms": 2.5},
                                   {"category": "Ofis"}, {"location": " "}, {"floor": 20, "n_floors": 10},
                                   {"has_repair": "maybe"}])
def test_bad_requests_rejected(service, patch):
    with pytest.raises(ValueError):
        service.predict({**GOOD, **patch})


def test_unknown_location_is_flagged_approximate(service):
    assert service.predict({**GOOD, "location": "Nowhere q."})["location_coordinate_approximate"] is True
