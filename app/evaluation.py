"""
No-show evaluation: time-based split, constant baseline, metrics, leakage guard.

All data used here is SYNTHETIC (data/bookings.csv, is_synthetic = True).

A no-show is a booking that does not reach COMPLETED, including late cancellations
(did_not_show = 1). The definition is fixed for the whole project.

    python -m app.evaluation
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bookings.csv"
LABEL = "did_not_show"
TRAIN_FRACTION = 0.75

# Only known after the appointment happens; using any of them as a feature is leakage.
FORBIDDEN_FEATURES = (
    "booking_status",
    "payment_status",
    "completed_at",
    "cancelled_at",
    "scheduled_end",
    LABEL,
)


def assert_no_leakage(features: list[str] | tuple[str, ...]) -> None:
    leaked = [f for f in features if f in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"post-appointment columns used as features: {leaked}")


def load_bookings(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["created_at"])
    return df.sort_values("created_at", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Earliest rows by created_at train, latest rows test. Never random."""
    df = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


def baseline_rate(train: pd.DataFrame) -> float:
    """Constant probability for every booking, from the training period only."""
    return float(train[LABEL].mean())


@lru_cache(maxsize=1)
def baseline_probability() -> float:
    train, _ = time_split(load_bookings())
    return baseline_rate(train)


def calibration_table(y_true, prob, bins: int = 10) -> list[dict]:
    """Predicted vs observed no-show rate per probability bin (empty bins skipped)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = idx == i
        if not mask.any():
            continue
        rows.append({
            "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
            "n": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
        })
    return rows


def evaluate(y_true, prob, threshold: float = 0.5) -> dict:
    """ROC-AUC, precision, recall at `threshold`, and a calibration table."""
    y = np.asarray(y_true)
    p = np.asarray(prob, dtype=float)
    flagged = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "threshold": threshold,
        "calibration": calibration_table(y, p),
    }


def run_baseline(df: pd.DataFrame | None = None) -> dict:
    df = load_bookings() if df is None else df
    train, test = time_split(df)
    rate = baseline_rate(train)
    prob = np.full(len(test), rate)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "split_after": str(train["created_at"].iloc[-1]),
        "overall_rate": float(df[LABEL].mean()),
        "train_rate": rate,
        "test_rate": float(test[LABEL].mean()),
        "metrics": evaluate(test[LABEL], prob),
    }


def main() -> None:
    r = run_baseline()
    m = r["metrics"]
    print("SYNTHETIC data: data/bookings.csv")
    print(f"train {r['train_rows']:,} rows, test {r['test_rows']:,} rows "
          f"(split after {r['split_after']})")
    print(f"overall no-show rate {r['overall_rate']:.2%}")
    print(f"train rate (the baseline) {r['train_rate']:.2%}, test rate {r['test_rate']:.2%}")
    print(f"ROC-AUC {m['roc_auc']:.3f}  precision {m['precision']:.3f}  "
          f"recall {m['recall']:.3f}  (threshold {m['threshold']})")
    print("calibration (bin, n, predicted, observed):")
    for row in m["calibration"]:
        print(f"  {row['bin']}  {row['n']:>6}  {row['predicted']:.3f}  {row['observed']:.3f}")


if __name__ == "__main__":
    main()
