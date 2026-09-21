"""
Wasaa Lifestyle — synthetic seed data.

One coherent world, shared by all ten use cases. Every table joins on the same
ids, so team 46's dashboard can genuinely aggregate what teams 37, 38, 40 and 41
produce. If each team invented its own ids, nothing would integrate.

EVERYTHING HERE IS SYNTHETIC. It is labelled as such in every file and must be
labelled as such in your report. Nobody loses marks for honest synthetic data.
Presenting it as real is a different matter.

Shapes follow the production Prisma schema — BookingStatus, PaymentStatus,
ServiceSlot and so on are the real enums and field names, so anything you build
against this drops onto the live API without renaming.

    python generate.py                 # default size
    python generate.py --households 1500 --months 18
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)          # fixed seed: same input, same data, always

# ── the world ────────────────────────────────────────────────────────────────

COUNTIES = {
    'Nairobi': 1.00, 'Kiambu': 0.88, 'Mombasa': 0.92, 'Nakuru': 0.80,
    'Kisumu': 0.84, 'Uasin Gishu': 0.76, 'Machakos': 0.78, 'Kakamega': 0.70,
}

# category -> (base price KES, typical duration mins, buys per household per month,
#              show bonus: added to the probability the household turns up)
#
# Categories are not equally missable. A burst pipe gets attended; a Saturday
# beauty appointment booked on a whim does not. Positive = more likely to show.
CATEGORIES = {
    'Home Cleaning':      (1800, 180, 2.20, -0.01),
    'Plumbing Repair':    (2500,  90, 0.30, +0.07),
    'Electrical Works':   (3000, 120, 0.25, +0.06),
    'Laundry':            (900,   60, 1.40, -0.03),
    'Cooking':            (1500, 150, 0.60, -0.02),
    'Childcare':          (2000, 300, 0.80, +0.05),
    'Tutoring':           (1200,  60, 1.10, -0.05),
    'Salon and Beauty':   (2200, 120, 0.90, -0.07),
    'Gardening':          (1600, 120, 0.35, -0.04),
    'Appliance Repair':   (2800,  90, 0.20, +0.03),
    'Moving and Errands': (3500, 240, 0.12, +0.04),
    'Security Install':   (5000, 180, 0.05, +0.05),
}

PAYMENT_METHODS = ['MPESA', 'CARD', 'WALLET']


def _rand_time_of_day() -> time:
    """Bookings cluster in the morning and late afternoon, not at 3am."""
    hour = int(RNG.choice(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        p=[.06, .11, .13, .11, .08, .05, .05, .07, .09, .11, .07, .05, .02],
    ))
    return time(hour, int(RNG.choice([0, 30])))


# ── households and providers ─────────────────────────────────────────────────

def make_households(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        county = str(RNG.choice(list(COUNTIES)))
        dependants = int(RNG.choice([0, 1, 2, 3, 4], p=[.28, .26, .24, .15, .07]))
        income = float(RNG.lognormal(11.0, 0.55) * COUNTIES[county])
        rows.append(dict(
            user_id=f'U{i:05d}',
            county=county,
            dependants=dependants,
            monthly_income_kes=round(income, 2),
            # How reliable this household is. Drives no-shows, and is NOT a column
            # you may use as a feature — it is the hidden truth behind the label.
            #
            # Deliberately two populations, not one bell curve: most households
            # nearly always turn up, and a small chronic segment repeatedly does
            # not. That segment is the whole commercial point of the model, and a
            # single normal distribution would hide it.
            hidden_reliability=float(np.clip(
                RNG.normal(0.975, 0.026) if RNG.random() > 0.13 else RNG.normal(0.62, 0.12),
                0.25, 0.999)),
            joined_at=(datetime(2024, 1, 1) + timedelta(days=int(RNG.integers(0, 540)))),
            is_synthetic=True,
        ))
    return pd.DataFrame(rows)


def make_providers(n: int) -> pd.DataFrame:
    rows = []
    cats = list(CATEGORIES)
    for i in range(n):
        cat = str(RNG.choice(cats))
        rows.append(dict(
            provider_id=f'P{i:04d}',
            business_name=f'{cat.split()[0]} Services {i:04d}',
            category=cat,
            county=str(RNG.choice(list(COUNTIES))),
            is_verified=bool(RNG.random() < 0.72),
            rating_average=(rating := round(float(np.clip(RNG.normal(4.3, 0.45), 2.6, 5.0)), 2)),
            total_completed_jobs=int(RNG.integers(0, 400)),
            # Hidden: how often this provider causes a failed job. Derived from the
            # public rating plus noise — a badly rated provider really is worse,
            # which is what makes rating_average worth feeding to a model.
            hidden_punctuality=float(np.clip(
                0.962 + 0.055 * (rating - 4.3) + RNG.normal(0, 0.012), 0.80, 0.999)),
            is_synthetic=True,
        ))
    return pd.DataFrame(rows)


def make_services(providers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in providers.itertuples():
        for k in range(int(RNG.integers(1, 4))):
            base, dur, _, _ = CATEGORIES[p.category]
            rows.append(dict(
                service_id=f'S{len(rows):05d}',
                provider_id=p.provider_id,
                category=p.category,
                name=f'{p.category} — package {k + 1}',
                base_price_kes=round(base * float(RNG.normal(1.0, 0.18)), 2),
                duration_minutes=dur,
                slot_capacity=int(RNG.choice([1, 1, 1, 2])),
                slots_enabled=True,
                is_synthetic=True,
            ))
    return pd.DataFrame(rows)


# ── bookings — the table team 41 needs ───────────────────────────────────────

def make_bookings(households, providers, services, months, start=datetime(2024, 1, 1)):
    """Bookings with real outcomes.

    The no-show label is driven by causes that are visible in the features, so the
    signal is genuinely learnable. In order of strength:

      lead time        booking three weeks out is forgotten; same-day rarely is
      first booking    a household's first ever booking is the riskiest
      prior no-shows   the single strongest signal, as it usually is
      time of day      very early and late slots get missed
      price            cheap jobs carry less commitment
      provider         some providers cause the failure themselves
    """
    # Plain lists, not DataFrame.sample — sampling a frame per booking is the
    # difference between four seconds and four minutes at this size.
    svc_by_provider: dict[str, list] = {}
    for s in services.itertuples():
        svc_by_provider.setdefault(s.provider_id, []).append(s)

    prov_all = list(providers.itertuples())
    prov_by_county: dict[str, list] = {}
    for pr in prov_all:
        prov_by_county.setdefault(pr.county, []).append(pr)

    rows = []
    history: dict[str, dict] = {u: {'n': 0, 'no_shows': 0} for u in households.user_id}
    # (service, date, HH:MM) -> how many bookings already sit in that slot. The
    # real scheduler refuses an over-full slot, so the seed data must too —
    # otherwise team 41 would be handed data their own rules say is impossible.
    slot_load: dict[tuple, int] = {}

    for h in households.itertuples():
        # A household books from its own county where it can.
        pool = prov_by_county.get(h.county) or prov_all
        rate = 0.55 + 0.30 * h.dependants + h.monthly_income_kes / 90_000
        for m in range(months):
            month_start = start + timedelta(days=30 * m)
            if month_start < h.joined_at:
                continue
            for _ in range(int(RNG.poisson(max(rate, 0.2)))):
                prov = pool[int(RNG.integers(0, len(pool)))]
                svcs = svc_by_provider.get(prov.provider_id)
                if not svcs:
                    continue
                svc = svcs[int(RNG.integers(0, len(svcs)))]

                created = month_start + timedelta(
                    days=float(RNG.uniform(0, 29)), hours=float(RNG.uniform(6, 21)))
                lead_days = float(np.clip(RNG.exponential(4.0), 0, 30))
                sched_day = created + timedelta(days=lead_days)
                sched = datetime.combine(sched_day.date(), _rand_time_of_day())
                if sched <= created:
                    sched = created + timedelta(hours=3)

                # Walk forward in 30-minute steps until a slot has room.
                cap = int(svc.slot_capacity)
                for _try in range(12):
                    key = (svc.service_id, sched.date(), sched.strftime('%H:%M'))
                    if slot_load.get(key, 0) < cap:
                        break
                    sched += timedelta(minutes=30)
                else:
                    continue            # that day is genuinely full — no booking
                slot_load[key] = slot_load.get(key, 0) + 1

                price = float(svc.base_price_kes) * float(RNG.normal(1.0, 0.08))
                hist = history[h.user_id]

                # ── probability the household turns up ───────────────────────
                p = h.hidden_reliability * prov.hidden_punctuality
                p += CATEGORIES[svc.category][3]             # some jobs are more missable
                p -= 0.0105 * lead_days                      # the dominant driver
                if hist['n'] == 0:
                    p -= 0.070                               # first-ever booking
                if hist['n']:
                    p -= 0.28 * (hist['no_shows'] / hist['n'])
                if sched.hour <= 8:
                    p -= 0.065
                elif sched.hour >= 17:
                    p -= 0.055
                p -= 0.030 * math.tanh((1500 - price) / 1500)  # cheap = less commitment
                if sched.weekday() >= 5:
                    p += 0.015
                p = float(np.clip(p, 0.12, 0.995))

                showed = RNG.random() < p
                if showed:
                    status, payment = 'COMPLETED', 'PAID'
                    completed_at = sched + timedelta(minutes=int(svc.duration_minutes))
                    cancelled_at = None
                elif RNG.random() < 0.42:
                    # cancelled, but too late to refill the slot — still a loss
                    status, payment = 'CANCELLED', 'REFUNDED'
                    completed_at, cancelled_at = None, sched - timedelta(
                        hours=float(RNG.uniform(0.5, 12)))
                else:
                    status = 'NO_SHOW'
                    # Most no-shows are charged in full; some providers agree to
                    # return part of the fee rather than lose the household.
                    payment = 'PAID' if RNG.random() < 0.78 else 'PARTIALLY_REFUNDED'
                    completed_at, cancelled_at = None, None

                hist['n'] += 1
                hist['no_shows'] += 0 if showed else 1

                rows.append(dict(
                    booking_id=f'B{len(rows):07d}',
                    user_id=h.user_id,
                    provider_id=prov.provider_id,
                    service_id=svc.service_id,
                    category=svc.category,
                    county=h.county,
                    created_at=created,
                    scheduled_start=sched,
                    scheduled_end=sched + timedelta(minutes=int(svc.duration_minutes)),
                    completed_at=completed_at,
                    cancelled_at=cancelled_at,
                    total_amount_kes=round(price, 2),
                    payment_method=str(RNG.choice(PAYMENT_METHODS, p=[.74, .14, .12])),
                    booking_status=status,
                    payment_status=payment,
                    is_synthetic=True,
                ))

    df = pd.DataFrame(rows).sort_values('created_at').reset_index(drop=True)

    # ── features that are knowable BEFORE the session, and nothing else ──────
    df['lead_time_hours'] = (
        (df.scheduled_start - df.created_at).dt.total_seconds() / 3600).round(2)
    df['scheduled_hour'] = df.scheduled_start.dt.hour
    df['scheduled_dayofweek'] = df.scheduled_start.dt.dayofweek
    df['is_weekend'] = (df.scheduled_dayofweek >= 5).astype(int)

    df = df.sort_values(['user_id', 'created_at'])
    g = df.groupby('user_id')
    df['prior_bookings'] = g.cumcount()
    df['is_first_booking'] = (df.prior_bookings == 0).astype(int)
    failed = df.booking_status.ne('COMPLETED').astype(int)
    df['prior_no_shows'] = (
        failed.groupby(df.user_id).cumsum() - failed).astype(int)
    df['prior_no_show_rate'] = (
        df.prior_no_shows / df.prior_bookings.replace(0, np.nan)).fillna(0).round(4)

    df = df.merge(
        providers[['provider_id', 'rating_average', 'is_verified']], on='provider_id', how='left')

    # the label
    df['did_not_show'] = (df.booking_status != 'COMPLETED').astype(int)

    return df.sort_values('created_at').reset_index(drop=True)


# ── slots, so team 41 also has the calendar half ─────────────────────────────

def make_slots(bookings: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    booked = (bookings.assign(date=bookings.scheduled_start.dt.normalize(),
                              start_time=bookings.scheduled_start.dt.strftime('%H:%M'))
                      .groupby(['service_id', 'date', 'start_time'])
                      .size().rename('booked_count').reset_index())
    cap = services.set_index('service_id').slot_capacity
    booked['capacity'] = booked.service_id.map(cap).fillna(1).astype(int)
    booked['slot_id'] = ['SL%07d' % i for i in range(len(booked))]
    booked['status'] = np.where(booked.booked_count >= booked.capacity, 'FULL', 'AVAILABLE')
    booked['is_blocked'] = False
    booked['is_synthetic'] = True
    return booked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--households', type=int, default=900)
    ap.add_argument('--providers', type=int, default=260)
    ap.add_argument('--months', type=int, default=24)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()

    import pathlib
    out = pathlib.Path(a.out)
    if not out.is_absolute():
        out = pathlib.Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)

    households = make_households(a.households)
    providers = make_providers(a.providers)
    services = make_services(providers)
    bookings = make_bookings(households, providers, services, a.months)
    slots = make_slots(bookings, services)

    # the hidden drivers never ship — they would leak the answer
    households.drop(columns=['hidden_reliability']).to_csv(out / 'households.csv', index=False)
    providers.drop(columns=['hidden_punctuality']).to_csv(out / 'providers.csv', index=False)
    services.to_csv(out / 'services.csv', index=False)
    bookings.to_csv(out / 'bookings.csv', index=False)
    slots.to_csv(out / 'slots.csv', index=False)

    rate = bookings.did_not_show.mean()
    print(f'households {len(households):>6,}   providers {len(providers):>5,}   '
          f'services {len(services):>5,}')
    print(f'bookings   {len(bookings):>6,}   slots     {len(slots):>5,}')
    print(f'\n{bookings.booking_status.value_counts().to_string()}')
    print(f'\nno-show / failed rate: {rate:.1%}')
    print(f'date range: {bookings.created_at.min():%Y-%m-%d} -> {bookings.created_at.max():%Y-%m-%d}')
    print(f'\nwritten to {out.resolve()}/')


if __name__ == '__main__':
    main()
