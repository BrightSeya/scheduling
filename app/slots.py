"""
Slot generation — U-CS 41, part one.

Turning a provider's weekly availability into concrete bookable slots.

This half is DETERMINISTIC. Given the same availability and the same date range you
must get exactly the same slots, every time. No randomness, no model, no guessing.
Get this right before you touch the no-show prediction, because a clever model on a
calendar that double-books is worth nothing.

Two settings do most of the work:

  slotDurationMinutes   how long one bookable slot is
  bufferMinutes         dead time AFTER each slot, so a provider can travel between
                        jobs. Ignore this and you generate schedules that are
                        physically impossible to honour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta


@dataclass(frozen=True)
class Availability:
    """A repeating weekly window. dayOfWeek: Monday=0 ... Sunday=6."""
    dayOfWeek: int
    startTime: str          # "09:00"
    endTime: str            # "17:00"
    isAvailable: bool = True

    def window(self, on: date) -> tuple[datetime, datetime]:
        s = time.fromisoformat(self.startTime)
        e = time.fromisoformat(self.endTime)
        return datetime.combine(on, s), datetime.combine(on, e)


@dataclass
class Slot:
    """One bookable slot. `bookedCount` never exceeds `capacity` — that is the invariant."""
    serviceId: str
    date: str               # "2026-10-02"
    startTime: str          # "09:00"
    endTime: str            # "10:00"
    capacity: int = 1
    bookedCount: int = 0
    isBlocked: bool = False
    blockReason: str | None = None
    generatedFrom: str = "availability"

    @property
    def is_full(self) -> bool:
        return self.bookedCount >= self.capacity

    @property
    def is_bookable(self) -> bool:
        return not self.isBlocked and not self.is_full

    def key(self) -> str:
        return f"{self.serviceId}|{self.date}|{self.startTime}"

    def to_dict(self) -> dict:
        return {
            "serviceId": self.serviceId,
            "date": self.date,
            "startTime": self.startTime,
            "endTime": self.endTime,
            "capacity": self.capacity,
            "bookedCount": self.bookedCount,
            "isBlocked": self.isBlocked,
            "blockReason": self.blockReason,
            "generatedFrom": self.generatedFrom,
        }


@dataclass
class ServiceConfig:
    serviceId: str
    slotDurationMinutes: int = 60
    bufferMinutes: int = 0
    slotCapacity: int = 1
    availability: list[Availability] = field(default_factory=list)


def generate_slots(cfg: ServiceConfig, start: date, end: date) -> list[Slot]:
    """Materialise every bookable slot between `start` and `end`, inclusive.

    Why materialise rather than compute on read? Because a slot has state — how many
    people have booked it. You cannot hold that in a function that recomputes from
    availability each time. Generating on read is the single most common way this
    use case goes wrong: two readers compute the same free slot and both book it.
    """
    if cfg.slotDurationMinutes <= 0:
        raise ValueError("slotDurationMinutes must be positive")
    if cfg.bufferMinutes < 0:
        raise ValueError("bufferMinutes cannot be negative")
    if start > end:
        raise ValueError("start date is after end date")

    by_day: dict[int, list[Availability]] = {}
    for a in cfg.availability:
        if a.isAvailable:
            by_day.setdefault(a.dayOfWeek, []).append(a)

    step = timedelta(minutes=cfg.slotDurationMinutes)
    buffer = timedelta(minutes=cfg.bufferMinutes)

    slots: list[Slot] = []
    day = start
    while day <= end:
        for a in by_day.get(day.weekday(), []):
            cursor, window_end = a.window(day)
            # A slot only exists if it fits ENTIRELY inside the window.
            # A half-hour left at the end of the day is not a bookable hour.
            while cursor + step <= window_end:
                slot_end = cursor + step
                slots.append(Slot(
                    serviceId=cfg.serviceId,
                    date=day.isoformat(),
                    startTime=cursor.strftime("%H:%M"),
                    endTime=slot_end.strftime("%H:%M"),
                    capacity=cfg.slotCapacity,
                ))
                # The buffer is consumed AFTER the slot, so the next one starts later.
                cursor = slot_end + buffer
        day += timedelta(days=1)

    return slots
