# Data dictionary

Field names and enum values match the production Prisma schema of the Wasaa Lifestyle
backend, so anything you build against these CSVs also works against the live API.

## bookings.csv — the training table

### Identity and join keys

| column | type | notes |
|---|---|---|
| `booking_id` | string | `B0000000`, unique |
| `user_id` | string | joins to `households.csv` |
| `provider_id` | string | joins to `providers.csv` |
| `service_id` | string | joins to `services.csv` |
| `category` | string | one of 12 service categories |
| `county` | string | the household's county, copied here for convenience |

### Timestamps

| column | type | notes |
|---|---|---|
| `created_at` | ISO datetime | when the booking was made — **split on this** |
| `scheduled_start` | ISO datetime | when the job was due to start |
| `scheduled_end` | ISO datetime | **leaky** — derived after the fact |
| `completed_at` | ISO datetime / empty | **leaky** — only set when the job happened |
| `cancelled_at` | ISO datetime / empty | **leaky** — only set when it was cancelled |

### Money

| column | type | notes |
|---|---|---|
| `total_amount_kes` | float | what the household agreed to pay |
| `payment_method` | enum | `MPESA` · `CARD` · `WALLET` |
| `payment_status` | enum | **leaky** — `PAID` · `REFUNDED` · `PARTIALLY_REFUNDED` |

### Outcome

| column | type | notes |
|---|---|---|
| `booking_status` | enum | **leaky** — `COMPLETED` · `NO_SHOW` · `CANCELLED` |
| `did_not_show` | 0 / 1 | **the label.** 1 when `booking_status != COMPLETED` |

`did_not_show` covers both a true no-show and a late cancellation, because both leave the
provider with an empty slot they cannot refill. If you only want strict no-shows, filter on
`booking_status == 'NO_SHOW'` — but say which definition you used, and keep it the same
throughout.

### Engineered features — safe to use

All ten are knowable at the moment the booking is created.

| column | type | notes |
|---|---|---|
| `lead_time_hours` | float | `scheduled_start - created_at`. The strongest feature you own. |
| `scheduled_hour` | int 0–23 | hour of the appointment |
| `scheduled_dayofweek` | int 0–6 | Monday = 0 |
| `is_weekend` | 0 / 1 | Saturday or Sunday |
| `prior_bookings` | int | how many this household made before this one |
| `is_first_booking` | 0 / 1 | `prior_bookings == 0` |
| `prior_no_shows` | int | how many of those failed |
| `prior_no_show_rate` | float 0–1 | `prior_no_shows / prior_bookings`, 0 when new |
| `rating_average` | float 2.6–5.0 | the provider's public rating |
| `is_verified` | bool | whether the provider is verified |

The four `prior_*` columns are **expanding windows** — each row only sees the rows before it.
That is deliberate. If you build your own history features, do the same, or you will leak.

---

## slots.csv — the calendar

| column | type | notes |
|---|---|---|
| `slot_id` | string | `SL0000000` |
| `service_id` | string | joins to `services.csv` |
| `date` | ISO date | |
| `start_time` | `HH:MM` | slots sit on the hour and half-hour |
| `capacity` | int | how many bookings this slot can hold |
| `booked_count` | int | how many it holds — never exceeds `capacity` |
| `status` | enum | `AVAILABLE` · `FULL` |
| `is_blocked` | bool | always false in this set |

---

## households.csv — the customer side

| column | type | notes |
|---|---|---|
| `user_id` | string | |
| `county` | string | 8 counties |
| `dependants` | int 0–4 | household members covered by the account |
| `monthly_income_kes` | float | log-normal, scaled by county |
| `joined_at` | ISO datetime | no bookings exist before this date |

---

## providers.csv — the supply side

| column | type | notes |
|---|---|---|
| `provider_id` | string | |
| `business_name` | string | |
| `category` | string | providers serve one category |
| `county` | string | |
| `is_verified` | bool | |
| `rating_average` | float 2.6–5.0 | correlates with real punctuality — a useful feature |
| `total_completed_jobs` | int | |

---

## services.csv — what is on sale

| column | type | notes |
|---|---|---|
| `service_id` | string | |
| `provider_id` | string | |
| `category`, `name` | string | |
| `base_price_kes` | float | |
| `duration_minutes` | int | |
| `slot_capacity` | int 1–2 | |
| `slots_enabled` | bool | always true here |

---

## One column appears in every file

`is_synthetic` is `True` on every row of every table. It is there so that if this data ever
touches a real database, nobody has to guess which rows were invented.
