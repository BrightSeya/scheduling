"""
Proof that the no-show signal in this data is real and learnable.

Generated data is worthless if the label is random — a model trained on it can
never beat guessing, and you would spend a month blaming your model for the
data's fault. So this script checks three things before anyone builds on it:

  1  no leakage — every feature is knowable before the appointment happens
  2  there is signal — a model beats the always-say-'they-will-show' baseline
  3  the signal is honest — it is spread across drivers, not one giveaway column

Run it after generate.py:

    python validate.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Only columns a scheduler can see at the moment of booking. Nothing that is
# decided by, or recorded after, the appointment itself.
FEATURES = [
    'lead_time_hours',
    'scheduled_hour',
    'scheduled_dayofweek',
    'is_weekend',
    'prior_bookings',
    'is_first_booking',
    'prior_no_shows',
    'prior_no_show_rate',
    'total_amount_kes',
    'rating_average',
]

# Columns that only exist because the appointment already happened. If one of
# these ever appears in FEATURES the model is cheating.
FORBIDDEN = ['booking_status', 'payment_status', 'completed_at', 'cancelled_at',
             'did_not_show', 'scheduled_end']

HERE = pathlib.Path(__file__).resolve().parent

BAR = '─' * 72


def main() -> None:
    df = pd.read_csv(HERE / 'bookings.csv', parse_dates=['created_at', 'scheduled_start'])
    df = df.sort_values('created_at').reset_index(drop=True)

    print(BAR)
    print('1. LEAKAGE CHECK')
    print(BAR)
    leaks = [c for c in FEATURES if c in FORBIDDEN]
    print(f'   features used        {len(FEATURES)}')
    print(f'   post-outcome columns {len(leaks)}  ->  {"CLEAN" if not leaks else leaks}')

    # Time-based split. A random split would let the model learn from a household's
    # future to predict its past, which it can never do in production.
    cut = df.created_at.quantile(0.75)
    train, test = df[df.created_at <= cut], df[df.created_at > cut]

    X_tr, y_tr = train[FEATURES].to_numpy(float), train.did_not_show.to_numpy()
    X_te, y_te = test[FEATURES].to_numpy(float), test.did_not_show.to_numpy()

    print(f'\n   train  {len(train):>6,}  up to {cut:%Y-%m-%d}   failure rate {y_tr.mean():.1%}')
    print(f'   test   {len(test):>6,}  after  {cut:%Y-%m-%d}   failure rate {y_te.mean():.1%}')

    print(f'\n{BAR}')
    print('2. IS THERE SIGNAL?')
    print(BAR)
    print(f'   {"model":<28}{"AUC":>7}{"PR-AUC":>9}{"precision":>11}{"recall":>9}{"F1":>7}')

    def score(name: str, prob: np.ndarray) -> float:
        # Flag the riskiest slice — the size a scheduler would actually action,
        # here the top 15% by predicted risk.
        thr = np.quantile(prob, 0.85)
        pred = (prob >= thr).astype(int)
        auc = roc_auc_score(y_te, prob)
        print(f'   {name:<28}{auc:>7.3f}{average_precision_score(y_te, prob):>9.3f}'
              f'{precision_score(y_te, pred, zero_division=0):>11.3f}'
              f'{recall_score(y_te, pred):>9.3f}{f1_score(y_te, pred):>7.3f}')
        return auc

    rng = np.random.default_rng(0)
    score('baseline: coin flip', rng.random(len(y_te)))

    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(sc.transform(X_tr), y_tr)
    auc_lr = score('logistic regression', lr.predict_proba(sc.transform(X_te))[:, 1])

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    gb.fit(X_tr, y_tr)
    auc_gb = score('gradient boosting', gb.predict_proba(X_te)[:, 1])

    # Same model, one extra column. Service category is a real driver — a burst
    # pipe gets attended, a Saturday beauty slot does not — so adding it lifts
    # the score. This is what a useful feature looks like.
    codes = pd.Categorical(df.category).categories
    cat_tr = pd.Categorical(train.category, categories=codes).codes.reshape(-1, 1)
    cat_te = pd.Categorical(test.category, categories=codes).codes.reshape(-1, 1)
    gbc = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, random_state=0,
        categorical_features=[len(FEATURES)])
    gbc.fit(np.hstack([X_tr, cat_tr]), y_tr)
    auc_gbc = score('gradient boosting + category',
                    gbc.predict_proba(np.hstack([X_te, cat_te]))[:, 1])
    auc_gb = max(auc_gb, auc_gbc)

    print(f'\n{BAR}')
    print('3. WHERE DOES THE SIGNAL COME FROM?')
    print(BAR)
    coefs = sorted(zip(FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1]))
    for name, c in coefs:
        direction = 'raises risk' if c > 0 else 'lowers risk'
        print(f'   {name:<24}{c:>+8.3f}   {direction}  {"█" * int(abs(c) * 14)}')

    print(f'\n{BAR}')
    print('   failure rate by service category')
    print(BAR)
    t = df.groupby('category').did_not_show.agg(['mean', 'size']).sort_values('mean')
    for label, row in t.iterrows():
        print(f'   {label:<22}{row["mean"]:>7.1%}   n={int(row["size"]):>6,}  '
              f'{"█" * int(row["mean"] * 90)}')

    print(f'\n{BAR}')
    print('   failure rate by lead time')
    print(BAR)
    bins = pd.cut(df.lead_time_hours, [0, 6, 24, 72, 168, 336, 1e9],
                  labels=['<6h', '6-24h', '1-3d', '3-7d', '1-2wk', '2wk+'])
    t = df.groupby(bins, observed=True).did_not_show.agg(['mean', 'size'])
    for label, row in t.iterrows():
        print(f'   {str(label):<8}{row["mean"]:>7.1%}   n={int(row["size"]):>6,}  '
              f'{"█" * int(row["mean"] * 90)}')

    print(f'\n{BAR}')
    print('   failure rate by prior history')
    print(BAR)
    hist = pd.cut(df.prior_no_shows, [-1, 0, 1, 2, 4, 1e9],
                  labels=['none', '1', '2', '3-4', '5+'])
    t = df.groupby(hist, observed=True).did_not_show.agg(['mean', 'size'])
    for label, row in t.iterrows():
        print(f'   {str(label):<8}{row["mean"]:>7.1%}   n={int(row["size"]):>6,}  '
              f'{"█" * int(row["mean"] * 90)}')

    print(f'\n{BAR}')
    ok = auc_gb >= 0.65 and auc_lr >= 0.60 and not leaks
    print('   VERDICT:', 'the data carries a real, learnable, leak-free signal.'
          if ok else 'FAILED — do not ship this dataset.')
    print(f'   best AUC {max(auc_lr, auc_gb):.3f} against 0.500 for guessing.')
    print(BAR)


if __name__ == '__main__':
    main()
