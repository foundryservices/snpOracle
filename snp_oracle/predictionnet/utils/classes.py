from datetime import datetime, timedelta

from snp_oracle.predictionnet.utils.timestamp import get_before, get_now, get_timezone, iso8601_to_datetime, round_minute_down


class MinerHistory:
    """This class is used to store miner predictions along with their timestamps.
    Allows for easy formatting, filtering, and lookup of predictions by timestamp.
    """

    def __init__(self, uid: int, timezone=get_timezone()):
        self.predictions = {}
        self.uid = uid
        self.timezone = timezone

    def add_prediction(self, timestamp, prediction: float):
        if isinstance(timestamp, str):
            timestamp = iso8601_to_datetime(timestamp)
        timestamp = round_minute_down(timestamp)
        if prediction is not None:
            self.predictions[timestamp] = prediction

    def clear_old_predictions(self):
        # deletes predictions older than 24 hours
        start_time = round_minute_down(get_now()) - timedelta(hours=24)
        filtered_pred_dict = {key: value for key, value in self.predictions.items() if start_time <= key}
        self.predictions = filtered_pred_dict

    def format_predictions(self, reference_timestamp=None, hours: int = 2, minutes: int = 0):
        if reference_timestamp is None:
            reference_timestamp = round_minute_down(get_now())
        if isinstance(reference_timestamp, str):
            reference_timestamp = iso8601_to_datetime(reference_timestamp)
        start_time = round_minute_down(reference_timestamp) - timedelta(hours=hours, minutes=minutes)
        filtered_pred_dict = {
            key: value for key, value in self.predictions.items() if start_time <= key < reference_timestamp
        }
        return filtered_pred_dict

    def get_relevant_timestamps(self, reference_timestamp: datetime):
        # returns a list of aligned timestamps
        # round down reference to nearest 5m
        round_down_now = round_minute_down(reference_timestamp)
        # get the timestamps for the past 12 epochs
        timestamps = [round_down_now - timedelta(minutes=5 * i) for i in range(12)]
        # remove any timestamps that are not in the dicts
        filtered_list = [item for item in timestamps if item in self.predictions.keys()]
        return filtered_list

    def latest_timestamp(self):
        if not self.predictions:
            return get_before(minutes=60)
        else:
            return max(self.predictions.keys())
