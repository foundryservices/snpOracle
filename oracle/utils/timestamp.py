from typing import List
from datetime import datetime, timedelta

from pytz import timezone
import pandas_market_calendars as mcal

###############################
#           GETTERS           #
###############################


def get_timezone() -> timezone:
    """
    Set the Global shared timezone for all timestamp manipulation
    """
    return timezone("UTC")


def get_now() -> datetime:
    """
    Get the current datetime
    """
    return datetime.now(get_timezone())


def get_before(minutes: int = 5) -> datetime:
    """
    Get the datetime x minutes before now
    """
    now = get_now()
    return now - timedelta(minutes=minutes)


def get_midnight() -> datetime:
    """
    Get the most recent instance of midnight
    """
    return get_now().replace(hour=0, minute=0, second=0, microsecond=0)


def get_posix() -> float:
    """
    Get the current POSIX time, seconds that have elapsed since Jan 1 1970
    """
    return datetime_to_posix(get_now())


def get_iso8601() -> str:
    """
    Get the current timestamp as a string, convenient for requests
    """
    return datetime_to_iso8601(get_now())


###############################
#         CONVERTERS          #
###############################


def datetime_to_posix(timestamp: datetime) -> float:
    """
    Convert datetime to seconds that have elapsed since Jan 1 1970
    """
    return timestamp.timestamp()


def datetime_to_iso8601(timestamp: datetime) -> str:
    """
    Convert datetime to iso 8601 string
    """
    return timestamp.isoformat()


def iso8601_to_datetime(timestamp: str) -> datetime:
    """
    Convert iso 8601 string to datetime
    """
    return datetime.fromisoformat(timestamp)


def posix_to_datetime(timestamp: float) -> datetime:
    """
    Convert seconds since Jan 1 1970 to datetime
    """
    return datetime.fromtimestamp(timestamp, tz=get_timezone())


###############################
#          FUNCTIONS          #
###############################


def elapsed_seconds(timestamp1: datetime, timestamp2: datetime) -> float:
    """
    Absolute number of seconds between two timestamps
    """
    return abs((timestamp1 - timestamp2).total_seconds())


def round_minute_down(timestamp: datetime, base: int = 5) -> datetime:
    """
    Round the timestamp down to the nearest 5 minutes

    Example:
        >>> timestamp = datetime.now(timezone("UTC"))
        >>> timestamp
        datetime.datetime(2024, 11, 14, 18, 18, 16, 719482, tzinfo=<UTC>)
        >>> round_minute_down(timestamp)
        datetime.datetime(2024, 11, 14, 18, 15, tzinfo=<UTC>)
    """
    # Round the minute down to the nearest multiple of `base`
    correct_minute: int = timestamp.minute // base * base
    return timestamp.replace(minute=correct_minute, second=0, microsecond=0)


def is_query_time(prediction_interval: int, timestamp: str, tolerance: int = 120) -> bool:
    """
    Tolerance - in seconds, how long to allow after epoch start
    prediction_interval - in minutes, how often to predict

    Notes:
        This function will be called multiple times
        First, check that we are in a new epoch
        Then, check if we already sent a request in the current epoch
    """
    now = get_now()
    provided_timestamp = iso8601_to_datetime(timestamp)

    # The provided timestamp is the last time a request was made. If this timestamp
    # is from the current epoch, we do not want to make a request. One way to check
    # this is by checking that `now` and `provided_timestamp` are more than `tolerance`
    # apart from each other. When true, this means the `provided_timestamp` is from
    # the previous epoch
    been_long_enough = elapsed_seconds(now, provided_timestamp) > tolerance

    # return false early if we already know it has not been long enough
    if not been_long_enough:
        return False

    # If it has been long enough, let's check the epoch start time
    midnight = get_midnight()
    sec_since_open = elapsed_seconds(now, midnight)

    # To check if this is a new epoch, compare the current timestamp
    # to the expected beginning of an epoch. If we are within `tolerance`
    # seconds of a new epoch, then we are willing to send a request
    sec_since_epoch_start = sec_since_open % (prediction_interval * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance

    # We know a request hasn't been sent yet, so simply return T/F based
    # on beginning of epoch
    return beginning_of_epoch


def align_timepoints(filtered_pred_dict, cm_data, cm_timestamps):
    """Takes in a dictionary of predictions and aligns them to a list of coinmetrics prices.

    Args:
        filtered_pred_dict (dict): {datetime: float} dictionary of predictions.
        cm_data (List[float]): price data from coinmetrics corresponding to the datetimes in cm_timestamps.
        cm_timestamps (List[datetime]): timestamps corresponding to the values in cm_data.


    Returns:
        aligned_pred_values (List[float]): The values in filtered_pred_dict with timestamp keys that match cm_timestamps.
        aligned_cm_data (List[float]): The values in cm_data where cm_timestamps matches the timestamps in filtered_pred_dict.
        aligned_timestamps (List[datetime]): The timestamps corresponding to the values in aligned_pred_values and aligned_cm_data.
    """
    if len(cm_data) != len(cm_timestamps):
        raise ValueError("cm_data and cm_timepoints must be of the same length.")

    aligned_pred_values = []
    aligned_cm_data = []
    aligned_timestamps = []

    # Convert cm_timepoints to a set for faster lookup
    cm_timestamps_set = set(cm_timestamps)
    # Loop through filtered_pred_dict to find matching datetime keys
    for timestamp, pred_value in filtered_pred_dict.items():
        if timestamp in cm_timestamps_set:
            # Find the index in cm_timepoints to get corresponding cm_data
            index = cm_timestamps.index(timestamp)
            aligned_pred_values.append(pred_value)
            aligned_timestamps.append(timestamp)
            aligned_cm_data.append(cm_data[index])
    return aligned_pred_values, aligned_cm_data, aligned_timestamps


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

def get_mature_timestamps(timestamp, interval: int =5, N_TIMEPOINTS: int = 6) -> List[datetime]:
    mature_timestamps = [timestamp + timedelta(minutes=interval*(i+1)) for i in range(N_TIMEPOINTS)]
    return mature_timestamps

