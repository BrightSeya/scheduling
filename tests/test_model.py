"""Validation split, threshold and top-k helpers, and the model comparison. Data is synthetic."""
import pytest

from app.evaluation import (LABEL, best_f1_threshold, load_bookings,
                            three_way_split, top_k_metrics)
from app.model import logistic_model, score_model


def test_three_way_split_is_chronological_and_keeps_the_test_set():
    fit, val, test = three_way_split(load_bookings())
    assert len(fit) + len(val) == 15999
    assert len(test) == 5334
    assert fit["created_at"].max() <= val["created_at"].min()
    assert val["created_at"].max() <= test["created_at"].min()


def test_best_f1_threshold_separates_a_clean_split():
    assert best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.8)


def test_top_k_flags_the_riskiest_fraction():
    out = top_k_metrics([1, 0, 0, 0], [0.9, 0.8, 0.1, 0.1], fraction=0.5)
    assert out["flagged"] == 2
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(1.0)


def test_logistic_model_beats_chance_on_the_held_out_test_set():
    fit, val, test = three_way_split(load_bookings())
    row = score_model("logistic regression", logistic_model(), fit, val, test)
    assert row["roc_auc"] > 0.70
    assert row["top_precision"] > test[LABEL].mean()
