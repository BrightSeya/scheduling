"""Slot generation and conflict detection."""
from datetime import date

import pytest

from app.booking import SlotStore
from app.conflicts import detect
from app.slots import Availability, ServiceConfig, generate_slots


def cfg(**kw) -> ServiceConfig:
    base = dict(
        serviceId="svc_1",
        slotDurationMinutes=60,
        bufferMinutes=0,
        slotCapacity=1,
        availability=[Availability(dayOfWeek=3, startTime="09:00", endTime="12:00")],
    )
    base.update(kw)
    return ServiceConfig(**base)


# ── generation ───────────────────────────────────────────────────────────────
def test_generates_whole_slots_only():
    """09:00-12:00 in 60 min slots is three slots, not three and a bit."""
    slots = generate_slots(cfg(), date(2026, 10, 1), date(2026, 10, 2))
    assert [s.startTime for s in slots] == ["09:00", "10:00", "11:00"]


def test_partial_slot_at_the_end_is_not_generated():
    """09:00-12:00 in 90 min slots fits two. The last 30 min is not bookable."""
    slots = generate_slots(cfg(slotDurationMinutes=90), date(2026, 10, 1), date(2026, 10, 2))
    assert [s.startTime for s in slots] == ["09:00", "10:30"]


def test_buffer_pushes_the_next_slot_later():
    """With a 30 min buffer the next job cannot start the moment the last one ends."""
    slots = generate_slots(cfg(bufferMinutes=30), date(2026, 10, 1), date(2026, 10, 2))
    assert [s.startTime for s in slots] == ["09:00", "10:30"]


def test_only_the_right_weekday_is_used():
    slots = generate_slots(cfg(), date(2026, 10, 1), date(2026, 10, 31))
    assert {s.date for s in slots} == {"2026-10-01", "2026-10-08",
                                       "2026-10-15", "2026-10-22", "2026-10-29"}


def test_generation_is_deterministic():
    a = generate_slots(cfg(), date(2026, 10, 1), date(2026, 10, 31))
    b = generate_slots(cfg(), date(2026, 10, 1), date(2026, 10, 31))
    assert [s.key() for s in a] == [s.key() for s in b]


def test_unavailable_windows_produce_nothing():
    c = cfg(availability=[Availability(dayOfWeek=3, startTime="09:00",
                                       endTime="12:00", isAvailable=False)])
    assert generate_slots(c, date(2026, 10, 1), date(2026, 10, 31)) == []


@pytest.mark.parametrize("bad", [
    dict(slotDurationMinutes=0),
    dict(slotDurationMinutes=-30),
    dict(bufferMinutes=-10),
])
def test_bad_config_is_rejected_loudly(bad):
    with pytest.raises(ValueError):
        generate_slots(cfg(**bad), date(2026, 10, 1), date(2026, 10, 2))


def test_backwards_date_range_is_rejected():
    with pytest.raises(ValueError):
        generate_slots(cfg(), date(2026, 10, 31), date(2026, 10, 1))


# ── conflicts ────────────────────────────────────────────────────────────────
def loaded_store(**kw) -> SlotStore:
    store = SlotStore()
    store.load(generate_slots(cfg(**kw), date(2026, 10, 1), date(2026, 10, 2)))
    return store


def test_clean_calendar_has_no_conflicts():
    assert detect(loaded_store()) == []


def test_booking_a_blocked_slot_is_flagged():
    store = loaded_store()
    key = next(iter(store.slots))
    store.book(key, "user_a")
    store.slots[key].isBlocked = True
    store.slots[key].blockReason = "provider away"

    types = {f["type"] for f in detect(store)}
    assert "BLOCKED_SLOT" in types


def test_back_to_back_booked_jobs_breach_the_buffer():
    """Slots generated with no buffer, then checked against a 30 min travel rule."""
    store = loaded_store(bufferMinutes=0)
    keys = sorted(store.slots)[:2]
    for k in keys:
        store.book(k, "user_a")

    findings = detect(store, buffer_minutes=30)
    assert any(f["type"] == "BUFFER_VIOLATION" for f in findings)


def test_empty_adjacent_slots_do_not_breach_the_buffer():
    """A gap between two EMPTY slots harms nobody and must not be reported."""
    store = loaded_store(bufferMinutes=0)
    assert not any(f["type"] == "BUFFER_VIOLATION"
                   for f in detect(store, buffer_minutes=30))


def test_orphan_booking_is_flagged():
    store = loaded_store()
    key = next(iter(store.slots))
    store.book(key, "user_a")
    del store.slots[key]                      # slot regenerated away underneath it

    assert any(f["type"] == "ORPHAN_BOOKING" for f in detect(store))
