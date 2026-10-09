"""Readable reasons for a prediction, from the logistic model's coefficients. Data is synthetic."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from app.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _describe(feature: str, v: float) -> str:
    if feature == "lead_time_hours":
        return f"booked {v / 24:.0f} days ahead" if v >= 48 else f"booked {v:.0f} hours ahead"
    if feature == "prior_no_shows":
        return f"{int(v)} prior no-show{'' if int(v) == 1 else 's'}"
    if feature == "prior_no_show_rate":
        return f"{v:.0%} of earlier bookings not honoured"
    if feature == "prior_bookings":
        return f"{int(v)} earlier bookings"
    if feature == "is_first_booking":
        return "first booking on the platform" if v else "has booked before"
    if feature == "scheduled_hour":
        return f"appointment at {int(v):02d}:00"
    if feature == "scheduled_dayofweek":
        return f"appointment on a {_DAYS[int(v)]}"
    if feature == "is_weekend":
        return "weekend appointment" if v else "weekday appointment"
    if feature == "total_amount_kes":
        return f"booking value KES {v:,.0f}"
    if feature == "rating_average":
        return f"provider rated {v:.1f}"
    if feature == "is_verified":
        return "verified provider" if v else "unverified provider"
    return f"{feature} {v}"


def factors(pipeline: Pipeline, features: pd.DataFrame, top: int = 3) -> list[str]:
    """Top contributions to the logit for one booking, each tagged raises or lowers risk."""
    prep = pipeline.named_steps["prep"]
    coef = pipeline.named_steps["clf"].coef_[0]
    scaler = prep.named_transformers_["num"]
    categories = list(prep.named_transformers_["cat"].categories_[0])
    n = len(NUMERIC_FEATURES)

    row = features.iloc[[0]]
    scaled = scaler.transform(row[list(NUMERIC_FEATURES)])[0]
    contrib = {f: float(coef[i] * scaled[i]) for i, f in enumerate(NUMERIC_FEATURES)}

    category = row[CATEGORICAL_FEATURES[0]].iloc[0]
    cat_coef = coef[n:]
    # Relative to the average category, so it is comparable with the centred numeric terms.
    contrib["category"] = (float(cat_coef[categories.index(category)] - cat_coef.mean())
                           if category in categories else 0.0)

    # With no history, the history terms say nothing; "first booking" already covers it.
    if row["is_first_booking"].iloc[0]:
        for f in ("prior_bookings", "prior_no_shows", "prior_no_show_rate"):
            contrib.pop(f)

    out = []
    for f in sorted(contrib, key=lambda k: -abs(contrib[k]))[:top]:
        text = f"{category} service" if f == "category" else _describe(f, row[f].iloc[0])
        out.append(f"{text} ({'raises' if contrib[f] > 0 else 'lowers'} risk)")
    return out
