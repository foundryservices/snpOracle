from typing import List

import bittensor as bt
import numpy as np
import yfinance as yf

from oracle.utils.general import (
    convert_predictions_to_matrix, convert_prices_to_matrix, time_shift, pd_to_dict, rank_miners_by_epoch, rank
    )
from oracle.protocol import Challenge

################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    decay = 0.9
    N_TIMEPOINTS = self.N_TIMEPOINTS
    weights = np.linspace(0, len(self.available_uids) - 1, len(self.available_uids))
    decayed_weights = decay**weights
    symbol = "^GSPC"
    data = yf.download(tickers=symbol, period="5d", interval="5m", progress=False)
    price_dict = pd_to_dict(data)
    raw_deltas = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
    raw_correct_dir = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), False)
    ranks = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
    uid_map = {i: uid for i, uid in enumerate(self.available_uids)}
    for uid, response in enumerate(responses):
        current_miner = self.MinerHistory[uid_map[uid]]
        if (response.prediction is not None) and len(response.prediction) == N_TIMEPOINTS:
                self.MinerHistory[uid_map[uid]].add_prediction(response.timestamp, response.prediction)
        prediction_dict = current_miner.format_predictions(response.timestamp, minutes=self.prediction_interval*N_TIMEPOINTS)
        if not prediction_dict or len(prediction_dict) <= 1:
            raw_deltas[uid, :, :], raw_correct_dir[uid, :, :] = np.inf, False
        else:
            raw_deltas[uid, :, :], raw_correct_dir[uid, :, :] = calc_raw(prediction_dict, price_dict, response.timestamp, N_TIMEPOINTS=N_TIMEPOINTS)
    for t in range(N_TIMEPOINTS):
        ranks[:, :, t] = rank_miners_by_epoch(raw_deltas[:, :, t], raw_correct_dir[:, :, t])
    incentive_ranks = rank(-np.nanmean(np.nanmean(ranks, axis=2), axis=1))
    bt.logging.info(f"Incentive Ranks: {incentive_ranks}")
    rewards = decayed_weights[incentive_ranks]
    rewards[incentive_ranks == max(incentive_ranks)] = 0 # anyone tied for last place gets no reward
    bt.logging.info(f"Rewards: {rewards}")
    return rewards


def calc_raw(prediction_dict: dict, price_dict: dict, timestamp: str, N_TIMEPOINTS: int = 6 ) -> np.ndarray:
    """
    Calculate delta and whether the direction of prediction was correct for one miner over the past N_TIMEPOINTS epochs

    Args:
        uid: The miner uid taken from the metagraph for this response
        response: The synapse response from the miner containing the prediction
        close_price: The S&P500 close price for the current epoch (N_TIMEPOINTS+1 to calculate price direction)

    Returns:
        deltas: The absolute difference between the predicted price and the true price
            - numpy.ndarray: (N_TIMEPOINTS (epochs) x N_TIMEPOINTS (timepoints))
        correct_dirs: A boolean array for if the predicted direction matched the true direction
            - numpy.ndarray: (N_TIMEPOINTS (epochs) x N_TIMEPOINTS (timepoints))

    Notes:
         - first row is the current epoch with only one prediction from <self.prediction_interval> minutes ago
         - the final row is <N_TIMEPOINTS> epochs ago with  (N_TIMEPOINTS x self.prediction_interval = 30min) minute predictions for the current timepoint
         - the final column is the current timepoint with various prediction distances (5min, 10min,...)
    """
    prediction_array, _ = convert_predictions_to_matrix(prediction_dict, timestamp)
    close_price, close_price_array = convert_prices_to_matrix(price_dict, timestamp, N_TIMEPOINTS=N_TIMEPOINTS)
    prior_close_prices = close_price[0:-1].reshape(N_TIMEPOINTS, 1)
    pred_dir = prior_close_prices - prediction_array
    close_dir = prior_close_prices - close_price_array
    correct_dirs = (close_dir >= 0) == time_shift((pred_dir >= 0))
    correct_dirs[np.isnan(pred_dir)] = False
    correct_dirs[np.isnan(close_dir)] = False
    deltas = np.abs(close_price_array - time_shift(prediction_array))
    return deltas, correct_dirs


# from oracle.protocol import Challenge
# from oracle.utils.timestamp import round_minute_down, get_now
# from oracle.utils.classes import MinerHistory
# from oracle.utils.general import convert_predictions_to_matrix, convert_prices_to_matrix, time_shift, pd_to_dict, rank_miners_by_epoch
# from datetime import timedelta
# import numpy as np
# import yfinance as yf

# symbol = "^GSPC"
# data = yf.download(tickers=symbol, period="5d", interval="5m", progress=False)
# price_dict = pd_to_dict(data)

# now = round_minute_down(get_now())
# old = now - timedelta(minutes=25)
# responses = [Challenge(timestamp=now.isoformat()), Challenge(timestamp=now.isoformat())]
# responses[0].prediction = [6000, 6001, 6002, 6003, 6004, 6005]
# responses[1].prediction = [5000, 5001, 5002, 5003, 5004, 5005]
# mh = {0: MinerHistory(uid=0), 1: MinerHistory(uid=1)}
# mh[0].add_prediction(old, responses[0].prediction)
# mh[1].add_prediction(old, responses[1].prediction)


# N_TIMEPOINTS=6
# raw_deltas = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
# raw_correct_dir = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), False)
# ranks = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
# for uid, response in zip([0,1], responses):
#     current_miner = mh[uid]
#     mh[uid].add_prediction(response.timestamp, response.prediction)
#     prediction_dict = current_miner.format_predictions(response.timestamp, minutes=5*N_TIMEPOINTS)
#     raw_deltas[uid, :, :], raw_correct_dir[uid, :, :] = calc_raw(prediction_dict, price_dict, response.timestamp, N_TIMEPOINTS=N_TIMEPOINTS)

# for t in range(N_TIMEPOINTS):
#     ranks[:, :, t] = rank_miners_by_epoch(raw_deltas[:, :, t], raw_correct_dir[:, :, t])
# incentive_ranks = np.nanmean(np.nanmean(ranks, axis=2), axis=1).argsort().argsort()