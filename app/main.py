"""
U-CS 41 — Smart Scheduling

Three jobs, in this order:

  1. SLOT GENERATION    turn availability into concrete bookable slots
  2. CONFLICT DETECTION make it impossible to double-book, and find the subtler clashes
  3. NO-SHOW PREDICTION predict which bookings will not be honoured

1 and 2 are correctness work and come first. A clever model on a calendar that
double-books is worth nothing. 3 is the machine learning half and is still yours
to build.

  POST /slots/generate      in : { serviceId, from, to, slotDurationMinutes, ... }
                            out: { count, slots[] }
  POST /slots/book          in : { slotKey, userId }
                            out: { bookingId, slotKey, status }
  POST /conflicts/check     in : { bufferMinutes }
                            out: { conflicts[], clean }
  POST /noshow/predict      in : { bookingId }
                            out: { probability, factors[] }      <- YOUR MODEL GOES HERE

Do not change the /noshow/predict shape without agreeing it with: U-CS 46 (ops-metrics)
"""
from datetime import date

from fastapi import FastAPI, HTTPException

from app.booking import STORE, SlotBlocked, SlotNotFound, SlotTaken
from app.conflicts import detect
from app.schemas import Request, Response
from app.service import handle
from app.slots import Availability, ServiceConfig, generate_slots

app = FastAPI(title="U-CS 41 — scheduling", version="0.2.0")


@app.get("/health")
def health():
    """Hosting platforms call this to check the service is alive."""
    return {"status": "ok"}


# ── 1. slot generation ───────────────────────────────────────────────────────
@app.post("/slots/generate")
def slots_generate(body: dict):
    """Materialise slots from a weekly availability pattern.

    Deterministic: same input, same output, every time.
    """
    try:
        cfg = ServiceConfig(
            serviceId=body["serviceId"],
            slotDurationMinutes=int(body.get("slotDurationMinutes", 60)),
            bufferMinutes=int(body.get("bufferMinutes", 0)),
            slotCapacity=int(body.get("slotCapacity", 1)),
            availability=[Availability(**a) for a in body.get("availability", [])],
        )
        slots = generate_slots(
            cfg,
            date.fromisoformat(body["from"]),
            date.fromisoformat(body["to"]),
        )
    except KeyError as e:
        raise HTTPException(422, f"missing field: {e}")
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))

    STORE.load(slots)
    return {"count": len(slots), "slots": [s.to_dict() for s in slots]}


# ── 2. booking and conflicts ─────────────────────────────────────────────────
@app.post("/slots/book")
def slots_book(body: dict):
    """Book one seat. Concurrency-safe: two people cannot take the last one."""
    try:
        booking = STORE.book(body["slotKey"], body.get("userId", "anonymous"))
    except KeyError as e:
        raise HTTPException(422, f"missing field: {e}")
    except SlotNotFound:
        raise HTTPException(404, "no such slot")
    except SlotBlocked as e:
        raise HTTPException(409, str(e))
    except SlotTaken as e:
        raise HTTPException(409, str(e))
    return booking.to_dict()


@app.post("/conflicts/check")
def conflicts_check(body: dict | None = None):
    """Find every conflict in the current calendar. An empty list is the pass."""
    buffer_minutes = int((body or {}).get("bufferMinutes", 0))
    found = detect(STORE, buffer_minutes=buffer_minutes)
    ok, problems = STORE.invariants_hold()
    return {
        "clean": not found and ok,
        "conflicts": found,
        "invariantsHold": ok,
        "invariantProblems": problems,
    }


# ── 3. the machine learning half — still yours ───────────────────────────────
@app.post("/noshow/predict", response_model=Response)
def endpoint(body: Request) -> Response:
    return handle(body)
