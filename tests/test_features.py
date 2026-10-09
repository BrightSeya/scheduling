"""Feature builder. Data is synthetic."""
import pytest

from app.evaluation import FORBIDDEN_FEATURES, load_bookings
from app.features import (FEATURES, UnknownBooking, build_features,
                          features_for_booking)


def test_feature_list_has_no_post_appointment_columns():
    assert not set(FEATURES) & set(FORBIDDEN_FEATURES)


def test_build_features_has_exactly_the_feature_columns_and_no_gaps():
    out = build_features(load_bookings())
    assert list(out.columns) == list(FEATURES)
    assert not out.isna().any().any()
    assert len(out) == 21333


def test_known_booking_returns_one_row_matching_the_csv():
    df = load_bookings().set_index("booking_id")
    row = features_for_booking("B0008746")
    assert row.shape == (1, len(FEATURES))
    assert row["lead_time_hours"].iloc[0] == df.loc["B0008746", "lead_time_hours"]
    assert row["category"].iloc[0] == df.loc["B0008746", "category"]


def test_unknown_booking_raises():
    with pytest.raises(UnknownBooking):
        features_for_booking("bk_not_in_the_dataset")
