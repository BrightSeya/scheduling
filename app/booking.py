"""
Concurrency-safe booking — U-CS 41, part two. The important part.

THE PROBLEM. Two people tap "book" on the last remaining seat at the same moment.
Naively you read bookedCount (0), see room, and write bookedCount = 1. Twice. Now
two customers own one slot and a provider is double-booked.

This is not a rare edge case. It is exactly what happens when a popular slot opens.

THE FIX, here. A lock around read-check-write, so the sequence cannot interleave.

THE FIX, in production. The same idea at the database. Either

    SELECT ... FOR UPDATE            -- lock the row, then check, then write
    UPDATE slots SET bookedCount = bookedCount + 1
      WHERE id = $1 AND bookedCount < capacity   -- conditional, atomic

The second is neater: the database itself refuses the over-book, and you check how
many rows were affected. Zero means somebody beat you to it.

In-memory storage below is deliberate — this is a demo service. What matters is the
SHAPE of the fix, which is identical whichever store you use.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from app.slots import Slot


class SlotTaken(Exception):
    """Raised when the slot filled up before this request got the lock."""


class SlotNotFound(Exception):
    pass


class SlotBlocked(Exception):
    pass


@dataclass
class Booking:
    bookingId: str
    slotKey: str
    userId: str
    status: str = "CONFIRMED"

    def to_dict(self) -> dict:
        return {
            "bookingId": self.bookingId,
            "slotKey": self.slotKey,
            "userId": self.userId,
            "status": self.status,
        }


@dataclass
class SlotStore:
    """Holds slots and bookings. Every mutation goes through the lock."""
    slots: dict[str, Slot] = field(default_factory=dict)
    bookings: dict[str, Booking] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self, slots: list[Slot]) -> int:
        with self._lock:
            for s in slots:
                self.slots.setdefault(s.key(), s)
            return len(self.slots)

    def get(self, key: str) -> Slot | None:
        return self.slots.get(key)

    def book(self, slot_key: str, user_id: str) -> Booking:
        """Book one seat. Exactly one caller can win the last seat.

        Everything that reads and then writes bookedCount sits inside the lock.
        Move a single line out of it and the bug comes straight back.
        """
        with self._lock:
            slot = self.slots.get(slot_key)
            if slot is None:
                raise SlotNotFound(slot_key)
            if slot.isBlocked:
                raise SlotBlocked(slot.blockReason or "slot is blocked")
            if slot.is_full:
                raise SlotTaken(f"{slot_key} is full ({slot.bookedCount}/{slot.capacity})")

            slot.bookedCount += 1
            booking = Booking(
                bookingId=f"bk_{uuid.uuid4().hex[:10]}",
                slotKey=slot_key,
                userId=user_id,
            )
            self.bookings[booking.bookingId] = booking
            return booking

    def cancel(self, booking_id: str) -> Booking:
        """Cancelling frees the seat. Cancelling twice must NOT free two seats."""
        with self._lock:
            booking = self.bookings.get(booking_id)
            if booking is None:
                raise SlotNotFound(booking_id)
            if booking.status == "CANCELLED":
                return booking          # idempotent on purpose
            booking.status = "CANCELLED"
            slot = self.slots.get(booking.slotKey)
            if slot and slot.bookedCount > 0:
                slot.bookedCount -= 1
            return booking

    def invariants_hold(self) -> tuple[bool, list[str]]:
        """bookedCount must never exceed capacity, and must match live bookings.

        Worth running in a test after any concurrency exercise.
        """
        problems: list[str] = []
        with self._lock:
            live: dict[str, int] = {}
            for b in self.bookings.values():
                if b.status != "CANCELLED":
                    live[b.slotKey] = live.get(b.slotKey, 0) + 1
            for key, slot in self.slots.items():
                if slot.bookedCount > slot.capacity:
                    problems.append(
                        f"{key}: OVERBOOKED {slot.bookedCount}/{slot.capacity}")
                if slot.bookedCount != live.get(key, 0):
                    problems.append(
                        f"{key}: count {slot.bookedCount} != {live.get(key, 0)} live bookings")
        return (not problems), problems


STORE = SlotStore()
