"""
THE TEST THAT MATTERS.

Fire many simultaneous requests at a slot with one seat. Exactly one must win.

Run it before and after any change to booking.py. If you ever move a line out of
the lock, this test is what tells you — not a user, in week nine.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from app.booking import SlotStore, SlotTaken
from app.slots import Availability, ServiceConfig, generate_slots


def one_seat_store() -> tuple[SlotStore, str]:
    """A store holding exactly one slot with exactly one seat."""
    cfg = ServiceConfig(
        serviceId="svc_1",
        slotDurationMinutes=60,
        slotCapacity=1,
        availability=[Availability(dayOfWeek=3, startTime="09:00", endTime="10:00")],
    )
    slots = generate_slots(cfg, date(2026, 10, 1), date(2026, 10, 2))  # Thu 2 Oct
    assert len(slots) == 1, "fixture should produce exactly one slot"
    store = SlotStore()
    store.load(slots)
    return store, slots[0].key()


@pytest.mark.parametrize("attempts", [2, 10, 50])
def test_only_one_booking_wins_the_last_seat(attempts: int):
    store, key = one_seat_store()

    def try_book(i: int):
        try:
            return ("won", store.book(key, f"user_{i}"))
        except SlotTaken:
            return ("lost", None)

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        results = list(pool.map(try_book, range(attempts)))

    winners = [r for r in results if r[0] == "won"]
    assert len(winners) == 1, f"{len(winners)} bookings won the same seat"
    assert store.get(key).bookedCount == 1


def test_capacity_is_never_exceeded_under_load():
    """Capacity 3, 40 simultaneous attempts. Exactly 3 win."""
    cfg = ServiceConfig(
        serviceId="svc_2",
        slotDurationMinutes=60,
        slotCapacity=3,
        availability=[Availability(dayOfWeek=3, startTime="09:00", endTime="10:00")],
    )
    slots = generate_slots(cfg, date(2026, 10, 1), date(2026, 10, 2))
    store = SlotStore()
    store.load(slots)
    key = slots[0].key()

    def try_book(i: int):
        try:
            store.book(key, f"user_{i}")
            return True
        except SlotTaken:
            return False

    with ThreadPoolExecutor(max_workers=40) as pool:
        wins = sum(pool.map(try_book, range(40)))

    assert wins == 3
    assert store.get(key).bookedCount == 3

    ok, problems = store.invariants_hold()
    assert ok, problems


def test_cancel_frees_exactly_one_seat_even_if_called_twice():
    """Cancelling twice must not free two seats. Idempotency, same as the wallet."""
    store, key = one_seat_store()
    booking = store.book(key, "user_a")
    assert store.get(key).bookedCount == 1

    store.cancel(booking.bookingId)
    store.cancel(booking.bookingId)          # again, on purpose
    assert store.get(key).bookedCount == 0

    ok, problems = store.invariants_hold()
    assert ok, problems

    # the seat is genuinely free again
    store.book(key, "user_b")
    assert store.get(key).bookedCount == 1
