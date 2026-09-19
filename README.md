# scheduling  —  U-CS 41

**Smart Scheduling** · Wasaa Lifestyle

Generate bookable slots correctly, and predict which bookings will not be honoured.

---

## Your contract

This is the shape the rest of the system expects. Agree any change with the teams below,
then tell Benjamin so the Team Integration Map gets reissued.

```
POST /noshow/predict
  in : { "bookingId": "..." }
  out: { "probability": 0.0, "factors": [] }
```

**Call it `scheduling` everywhere** — your repo name, your service URL, your route, and the
variable Benjamin sets when integrating you: `SCHEDULING_SERVICE_URL`

## Who you depend on

Nothing.

## Who depends on you

U-CS 46 (ops-metrics)

## Done when

Two people cannot book the last seat — prove it with a test — and your no-show model is calibrated.

---

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> — FastAPI gives you interactive API docs for free, so you can
test your endpoint without writing a client. Open `demo/index.html` for the mini app.

```bash
pytest -q                 # the contract test
docker build -t scheduling . && docker run -p 8000:8000 scheduling
```

## The order of work

1. **Baseline.** The dumbest thing that works, returned from `app/service.py`. Measure it and
   **write the number down.** You are not allowed to build a model before this exists.
2. **Evaluation.** Pick your metric. Split your data **by time**, not at random, and hold the
   last part back.
3. **Model.** Build the real thing. Show it beats the baseline.
4. **Integration.** Keep the contract identical while the inside changes completely. Deploy it,
   send Benjamin the URL.

## What you hand in

- This repo, on GitHub, with Benjamin added as a collaborator
- A live URL that answers
- The filled-in documents (templates are in the `documents/` folder of the skeleton repo)
- A README a stranger could run this from

## Rules

- No secrets in git. No `.env`, no keys. Commit `.env.example` instead.
- Label synthetic data as synthetic — in the code and in the report.
- Staging data only. Never production, never live M-Pesa.
- Tell Benjamin the same day if you find a real bug in the platform.

---

## What is already built, and what is yours

U-CS 41 is three jobs. The first two are correctness work and are done — a clever model on
a calendar that double-books is worth nothing. The third is yours.

| | Job | Status |
|---|---|---|
| 1 | **Slot generation** — availability into concrete bookable slots | done, `app/slots.py` |
| 2 | **Conflict detection** — double-booking, buffers, overlaps, orphans | done, `app/conflicts.py` + `app/booking.py` |
| 3 | **No-show prediction** | **yours** — `app/service.py` |

### Endpoints

```
POST /slots/generate    in : { serviceId, from, to, slotDurationMinutes, bufferMinutes, availability[] }
                        out: { count, slots[] }
POST /slots/book        in : { slotKey, userId }        concurrency-safe
                        out: { bookingId, slotKey, status }
POST /conflicts/check   in : { bufferMinutes }
                        out: { clean, conflicts[], invariantsHold }
POST /noshow/predict    in : { bookingId }              <- YOUR MODEL
                        out: { probability, factors[] }
```

The `/noshow/predict` shape is fixed — U-CS 46 (ops-metrics) depends on it.

### The test that matters

`tests/test_concurrency.py` fires 50 simultaneous requests at a slot with one seat and asserts
exactly one wins. Run it before and after any change to `app/booking.py`:

```bash
pytest tests/ -q          # 22 tests
```

It is not a decorative test. With the lock removed from `book()` it fails immediately.

### Your next steps

1. Fill in the **Concept Note** in `documents/`
2. Define **no-show** precisely — a booking that reached its start and never completed, or was
   cancelled inside a window you choose. Write the definition down in week 1 and do not change it
3. Build the **baseline**: predict the overall no-show rate for everyone. Measure it. Write the
   number down. You are not allowed a model until that number exists
4. Only then build the real model, using **pre-session features only**. If you use `cancelledAt`
   you have leaked the answer into the question
