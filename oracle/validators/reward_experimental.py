from typing import List

import numpy as np
import yfinance as yf

from oracle.utils.timestamp import get_mature_timestamps, align_timepoints, get_now, round_minute_down
from oracle.protocol import Challenge

################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    decay = 0.9
    weights = np.linspace(0, len(self.available_uids) - 1, len(self.available_uids))
    decayed_weights = decay**weights
    symbol = "^GSPC"
    data = yf.download(tickers=symbol, period="5d", interval="5m", progress=False)
    price_dict = pd_to_dict(data)
    ranks = np.full(len(self.available_uids), np.nan)
    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction)
        prediction_dict = current_miner.format_predictions(response.timestamp, minutes=self.prediction_interval*self.N_TIMEPOINTS)
        dir_acc = get_dir_acc(self, prediction_dict, price_dict) # 1 is perfect, 0 is bad
        delta_acc = get_delta_acc(self, prediction_dict, data) # 0 is perfect, inf is bad
    final_ranks = rank((rank(-dir_acc) + rank(delta_acc))/2 )
    rewards = decayed_weights[final_ranks]
    return rewards


def get_dir_acc(self, prediction_dict, price_dict):
    avg_dir_acc_for_epoch = []
    for key, value in prediction_dict.items():
        mature_timestamps = get_mature_timestamps(key, interval=self.prediction_interval, n_timepoints=self.N_TIMEPOINTS)
        relavent_prices, relavent_predictions, _ = align_timepoints(price_dict, value, [mature_timestamps, key])
        start_price = price_dict[key]
        both_increase = relavent_prices>start_price and relavent_predictions>start_price
        both_decrease = relavent_prices<start_price and relavent_predictions<start_price
        dir_acc = both_increase or both_decrease
        avg_dir_acc_for_epoch.append(np.mean(dir_acc))
    return np.mean(avg_dir_acc_for_epoch)


def get_delta_acc(self, prediction_dict, price_dict):
    deltas = []
    for key, value in prediction_dict.items():
        mature_timestamps = get_mature_timestamps(key, interval=self.prediction_interval, n_timepoints=self.N_TIMEPOINTS)
        relavent_prices, relavent_predictions, _ = align_timepoints(price_dict, value, mature_timestamps)
        deltas.append(np.abs(relavent_predictions - relavent_prices))
    return np.mean(deltas)


def pd_to_dict(data):
    price_dict = {}
    for i in range(len(data)):
        price_dict[data.index[i]] = data.iloc[i]['Close'].values.item()
    return price_dict


def rank(vector):
    if vector is None or len(vector) <= 1:
        return np.array([0])
    else:
        # Sort the array and get the indices that would sort it
        sorted_indices = np.argsort(vector)
        sorted_vector = vector[sorted_indices]
        # Create a mask for where each new unique value starts in the sorted array
        unique_mask = np.concatenate(([True], sorted_vector[1:] != sorted_vector[:-1]))
        # Use cumulative sum of the unique mask to get the ranks, then assign back in original order
        ranks = np.cumsum(unique_mask) - 1
        rank_vector = np.empty_like(vector, dtype=int)
        rank_vector[sorted_indices] = ranks
        return rank_vector


def convert_to_matrix(prediction_dict):
    now = round_minute_down(get_now())
    timestamps_to_pull = get_mature_timestamps(now, interval=-5)
    predictions = np.zeros((len(timestamps_to_pull), len(timestamps_to_pull)))
    for i, timestamp in enumerate(timestamps_to_pull):
        try:
            predictions[i,:] = prediction_dict[timestamp]
        except KeyError:
            predictions[i,:] = np.nan
    return predictions