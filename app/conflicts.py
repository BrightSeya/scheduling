"""
Conflict detection — U-CS 41, part three.

Double-booking is the obvious conflict. It is not the only one, and the others are
what make this use case interesting.

  OVERBOOKED        more bookings than capacity
  BUFFER_VIOLATION  two jobs back to back with no travel time between them
  BLOCKED_SLOT      a booking sits on a slot the provider later blocked
  OVERLAP           two slots for the same service overlap in time at all
  ORPHAN_BOOKING    a booking pointing at a slot that no longer exists

Each finding names the slots involved so it can be acted on, not just counted.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.booking import SlotStore
from app.slots import Slot


def _dt(slot: Slot, which: str) -> datetime:
    t = slot.startTime if which == "start" else slot.endTime
    return datetime.fromisoformat(f"{slot.date}T{t}:00")


def detect(store: SlotStore, buffer_minutes: int = 0) -> list[dict]:
    """Return every conflict found. An empty list is the only passing result."""
    findings: list[dict] = []
    slots = list(store.slots.values())

    # 1. capacity
    for s in slots:
        if s.bookedCount > s.capacity:
            findings.append({
                "type": "OVERBOOKED",
                "slot": s.key(),
                "detail": f"{s.bookedCount} bookings for capacity {s.capacity}",
                "severity": "critical",
            })

    # 2. bookings on blocked slots
    for s in slots:
        if s.isBlocked and s.bookedCount > 0:
            findings.append({
                "type": "BLOCKED_SLOT",
                "slot": s.key(),
                "detail": f"{s.bookedCount} booking(s) on a slot blocked: "
                          f"{s.blockReason or 'no reason given'}",
                "severity": "critical",
            })

    # 3. overlap and buffer, per service per day
    buffer = timedelta(minutes=buffer_minutes)
    grouped: dict[tuple[str, str], list[Slot]] = {}
    for s in slots:
        grouped.setdefault((s.serviceId, s.date), []).append(s)

    for (service_id, day), day_slots in grouped.items():
        day_slots.sort(key=lambda x: x.startTime)
        for a, b in zip(day_slots, day_slots[1:]):
            a_end, b_start = _dt(a, "end"), _dt(b, "start")

            if b_start < a_end:
                findings.append({
                    "type": "OVERLAP",
                    "slot": a.key(),
                    "other": b.key(),
                    "detail": f"{a.endTime} overlaps {b.startTime} on {day}",
                    "severity": "critical",
                })
                continue

            # Only a real problem if both are actually booked — a gap between two
            # empty slots harms nobody.
            if buffer and a.bookedCount > 0 and b.bookedCount > 0:
                if b_start - a_end < buffer:
                    gap = int((b_start - a_end).total_seconds() // 60)
                    findings.append({
                        "type": "BUFFER_VIOLATION",
                        "slot": a.key(),
                        "other": b.key(),
                        "detail": f"only {gap} min between booked jobs, "
                                  f"{buffer_minutes} min required for travel",
                        "severity": "warning",
                    })

    # 4. bookings pointing at slots that no longer exist
    for b in store.bookings.values():
        if b.status != "CANCELLED" and b.slotKey not in store.slots:
            findings.append({
                "type": "ORPHAN_BOOKING",
                "slot": b.slotKey,
                "detail": f"booking {b.bookingId} references a slot that is gone",
                "severity": "critical",
            })

    return findings
