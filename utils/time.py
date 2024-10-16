from datetime import datetime, timedelta

import pandas_market_calendars as mcal
from pytz import timezone


def market_is_open() -> bool:
    """
    This function checks if the NYSE is open and validators should send requests.
    The final valid time is 3:55 PM EST

    Returns:
        True if the NYSE is open and the current time is between 9:30 AM and (4:00 PM - self.INTERVAL)
        False otherwise

    Notes:
    ------
    Timezone is set to America/New_York

    """
    est = timezone("America/New_York")
    now = datetime.now(est)
    # Check if today is Monday through Friday
    if now.weekday() >= 5:  # 0 is Monday, 6 is Sunday
        return False
    # Check if the NYSE is open (i.e. not a holiday)
    result = mcal.get_calendar("NYSE").schedule(start_date=now, end_date=now)
    if result.empty:
        return False
    # Check if the current time is between 9:30 AM and 4:00 PM
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if not (start_time <= now <= end_time):
        return False
    # if all checks pass, return true
    return True


def is_query_time(prediction_interval, timestamp) -> bool:
    now_ts = datetime.now(timezone("America/New_York")).timestamp()
    open_ts = datetime.now(timezone("America/New_York")).replace(hour=9, minute=30, second=0, microsecond=0).timestamp()
    sec_since_open = now_ts - open_ts
    tolerance = 120  # in seconds, how long to allow after epoch start
    # if it is within 120 seconds of the start of the prediction epoch
    beginning_of_epoch = sec_since_open % (prediction_interval * 60) < tolerance
    been_long_enough = datetime.now(timezone("America/New_York")) - datetime.fromisoformat(timestamp) > timedelta(seconds=tolerance)
    result = beginning_of_epoch and been_long_enough
    return result
