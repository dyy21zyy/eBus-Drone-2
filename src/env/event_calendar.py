from __future__ import annotations

import heapq
from typing import Optional

from .event_types import Event


class EventCalendar:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def add(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def pop_next(self) -> Optional[Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek_next(self) -> Optional[Event]:
        if not self._heap:
            return None
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)
