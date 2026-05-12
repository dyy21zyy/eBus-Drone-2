from src.env.event_calendar import EventCalendar

def test_cal(): c=EventCalendar(); c.add(2,"b"); c.add(1,"a"); assert c.pop_next()==(1,"a")
