from typing import List

import numpy as np
import yfinance as yf

from snp_oracle.predictionnet.protocol import Challenge
from snp_oracle.predictionnet.utils.general import (
    convert_predictions_to_matrix,
    convert_prices_to_matrix,
    pd_to_dict,
    rank,
    rank_miners_by_epoch,
    time_shift,
)


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
        prediction_dict = current_miner.format_predictions(
            response.timestamp, minutes=self.prediction_interval * N_TIMEPOINTS
        )
        if not prediction_dict or len(prediction_dict) <= 1:
            raw_deltas[uid, :, :], raw_correct_dir[uid, :, :] = np.inf, False
        else:
            raw_deltas[uid, :, :], raw_correct_dir[uid, :, :] = calc_raw(
                prediction_dict,
                price_dict,
                response.timestamp,
                N_TIMEPOINTS=N_TIMEPOINTS,
            )
    for t in range(N_TIMEPOINTS):
        ranks[:, :, t] = rank_miners_by_epoch(raw_deltas[:, :, t], raw_correct_dir[:, :, t])
    incentive_ranks = rank(-np.nanmean(np.nanmean(ranks, axis=2), axis=1))
    rewards = decayed_weights[incentive_ranks]
    rewards[incentive_ranks == max(incentive_ranks)] = 0  # anyone tied for last place gets no reward
    return rewards


def calc_raw(prediction_dict: dict, price_dict: dict, timestamp: str, N_TIMEPOINTS: int = 6) -> np.ndarray:
    """
    Calculate delta and whether the direction of prediction was correct for one miner over the past N_TIMEPOINTS epochs
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
