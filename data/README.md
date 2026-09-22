# Dataset — no-show prediction

You asked three questions. Short answers first.

**Will the dataset be provided?** Yes — it is in this folder. You do not have to create one.

**What fields?** `bookings.csv` has 27 columns. The label is `did_not_show`. Ten features
are already engineered for you and listed in [`validate.py`](validate.py); everything else is raw.

**How many records?** 21,333 bookings across 900 households, 260 providers and 24 months.
If you want more, run `python generate.py --households 3000 --months 36`.

> **This data is synthetic.** It was generated, not collected. Say so in your report —
> nobody loses marks for honest synthetic data, and the alternative is a problem.

---

## Files

| file | rows | what it is |
|---|---|---|
| `bookings.csv` | 21,333 | one row per booking, with the outcome — **this is the one you train on** |
| `slots.csv` | 21,322 | the slot grid, with `capacity` and `booked_count` |
| `households.csv` | 900 | the customer side — county, dependants, income |
| `providers.csv` | 260 | the supply side — category, rating, verification |
| `services.csv` | 525 | what each provider sells, with price and duration |

Every table joins on shared ids — `user_id`, `provider_id`, `service_id`. The same ids run
through the other teams' data, so what you build here lines up with theirs.

Column-by-column notes are in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## Run it

```bash
pip install -r ../requirements.txt
python generate.py     # rebuild the CSVs (fixed seed — same data every time)
python validate.py     # prove the signal is real before you train on it
```

## What `validate.py` tells you

It is not a model you are meant to submit. It exists so you know the data is worth your
month, and so you have a number to beat.

```
model                           AUC   PR-AUC  precision   recall     F1
baseline: coin flip           0.500    0.208      0.204    0.147  0.171
logistic regression           0.738    0.496      0.554    0.399  0.464
gradient boosting             0.743    0.494      0.554    0.399  0.464
gradient boosting + category  0.750    0.497      0.536    0.386  0.449
```

**Beat 0.750 AUC and you have done something real.** Report AUC and precision/recall, not
accuracy — 78% of these bookings are honoured, so a model that predicts "everyone shows up"
scores 78% accuracy and is worth nothing.

## Two traps in this data

**Leakage.** `booking_status`, `payment_status`, `completed_at`, `cancelled_at` and
`scheduled_end` only exist *because* the appointment already happened. Feed any of them to
the model and you will see 0.99 AUC in your notebook and garbage in production.
`validate.py` checks this on every run.

**Time.** Split by date, not at random. `validate.py` trains on the first 75% of bookings by
`created_at` and tests on the last 25%. A random split lets the model see a household's
future while predicting its past — which it can never do on a live booking.

## What actually drives the label

You do not need this to build the model, but it is what the data encodes, and it is worth
knowing whether your model found it:

| driver | effect |
|---|---|
| prior no-show rate | strongest by far — 13% for a clean record, 37% after five |
| lead time | 18% booked same-day, 33% booked more than two weeks out |
| service category | 14% plumbing (urgent, attended), 30% salon and laundry (skippable) |
| first-ever booking | a household with no history is riskier than one with a good one |
| price | cheap jobs carry less commitment |
| provider rating | badly rated providers genuinely cause more failures |

## Wiring it to your contract

Your endpoint is `POST /noshow/predict` with `{ "bookingId": "..." }`. So at request time you
look the booking up, build the same ten features, and return the probability:

```python
{ "probability": 0.31, "factors": ["booked 19 days ahead", "2 prior no-shows"] }
```

The feature names in `bookings.csv` match the fields on the real `Booking` model in the Wasaa
Lifestyle backend, so the same code works against the live API later.
