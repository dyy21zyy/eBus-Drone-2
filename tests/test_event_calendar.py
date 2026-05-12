from src.env.event_calendar import EventCalendar
from src.env.event_types import Event


def test_event_calendar_orders_events():
    c = EventCalendar()
    c.add(Event(2, 0, 1, 1, True, True, True))
    c.add(Event(1, 0, 0, 0, False, True, False))
    assert c.pop_next().time == 1
