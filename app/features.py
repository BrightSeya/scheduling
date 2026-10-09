"""
Pre-appointment features for the no-show model. Data is SYNTHETIC.

Only columns known at the moment a booking is created. The prior_* columns in
data/bookings.csv are expanding windows, so each row only sees earlier bookings.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.evaluation import assert_no_leakage, load_bookings

NUMERIC_FEATURES = (
    "lead_time_hours",
    "scheduled_hour",
    "scheduled_dayofweek",
    "is_weekend",
    "prior_bookings",
    "is_first_booking",
    "prior_no_shows",
    "prior_no_show_rate",
    "total_amount_kes",
    "rating_average",
    "is_verified",
)
CATEGORICAL_FEATURES = ("category",)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

assert_no_leakage(FEATURES)


class UnknownBooking(KeyError):
    """No booking with this id in the dataset."""


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(FEATURES)].copy()
    out["is_verified"] = out["is_verified"].astype(int)
    return out


@lru_cache(maxsize=1)
def _bookings_by_id() -> pd.DataFrame:
    return load_bookings().set_index("booking_id")


def features_for_booking(booking_id: str) -> pd.DataFrame:
    """One-row feature frame for a booking id; raises UnknownBooking if absent."""
    bookings = _bookings_by_id()
    if booking_id not in bookings.index:
        raise UnknownBooking(booking_id)
    return build_features(bookings.loc[[booking_id]])
