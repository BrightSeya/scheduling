"""
No-show model: fit on early data, tune the threshold on validation, score the test set once.
Data is SYNTHETIC.

    python -m app.model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.evaluation import (LABEL, baseline_rate, best_f1_threshold, evaluate,
                            load_bookings, three_way_split, top_k_metrics)
from app.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features

TARGET_AUC = 0.75
SEED = 0


def logistic_model() -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), list(NUMERIC_FEATURES)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def gradient_boosting_model() -> Pipeline:
    prep = ColumnTransformer([
        ("num", "passthrough", list(NUMERIC_FEATURES)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         list(CATEGORICAL_FEATURES)),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, random_state=SEED,
        categorical_features=[len(NUMERIC_FEATURES)])
    return Pipeline([("prep", prep), ("clf", clf)])


def score_model(name: str, model: Pipeline, fit: pd.DataFrame,
                val: pd.DataFrame, test: pd.DataFrame) -> dict:
    model.fit(build_features(fit), fit[LABEL].to_numpy())
    threshold = best_f1_threshold(
        val[LABEL], model.predict_proba(build_features(val))[:, 1])
    prob = model.predict_proba(build_features(test))[:, 1]
    return _row(name, test[LABEL], prob, threshold)


def _row(name: str, y_true, prob, threshold: float) -> dict:
    m = evaluate(y_true, prob, threshold)
    top = top_k_metrics(y_true, prob)
    return {
        "name": name,
        "roc_auc": m["roc_auc"],
        "threshold": threshold,
        "precision": m["precision"],
        "recall": m["recall"],
        "top_precision": top["precision"],
        "top_recall": top["recall"],
    }


def compare(df: pd.DataFrame | None = None) -> list[dict]:
    df = load_bookings() if df is None else df
    fit, val, test = three_way_split(df)
    # Same training-period rate as the baseline in app/evaluation.py (fit + validation).
    rate = baseline_rate(pd.concat([fit, val]))
    rows = [_row("baseline (constant)", test[LABEL], np.full(len(test), rate), rate)]
    rows.append(score_model("logistic regression", logistic_model(), fit, val, test))
    rows.append(score_model("gradient boosting", gradient_boosting_model(), fit, val, test))
    return rows


def main() -> None:
    print("SYNTHETIC data: data/bookings.csv")
    print(f"{'model':<22}{'ROC-AUC':>8}{'thr':>7}{'prec':>7}{'recall':>8}"
          f"{'top15 P':>9}{'top15 R':>9}")
    rows = compare()
    for r in rows:
        print(f"{r['name']:<22}{r['roc_auc']:>8.3f}{r['threshold']:>7.2f}"
              f"{r['precision']:>7.3f}{r['recall']:>8.3f}"
              f"{r['top_precision']:>9.3f}{r['top_recall']:>9.3f}")
    best = max(r["roc_auc"] for r in rows[1:])
    print(f"\nbest model ROC-AUC {best:.3f}  target {TARGET_AUC:.3f}  "
          f"{'reached' if best >= TARGET_AUC else 'not reached'}")


if __name__ == "__main__":
    main()
