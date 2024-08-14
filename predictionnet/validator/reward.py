# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from typing import List
import bittensor as bt
from predictionnet.protocol import Challenge
import time
from datetime import datetime, timedelta
import yfinance as yf
from pytz import timezone
import numpy as np


################################################################################
#                              Helper Functions                                #
################################################################################
def calc_raw(self, uid, response: Challenge, close_price: float):
    """
    Calculate delta and whether the direction of prediction was correct for a single response

    Args:
        uid: The miner uid taken from the metagraph for this response 
        response: The synapse response from the miner containing the prediction
        close_price: The S&P500 close price for the current epoch

    Returns:
        deltas: The absolute difference between the predicted price and the true price
            - numpy.ndarray: (N_TIMEPOINTS x N_TIMEPOINTS)
        correct_dirs: A boolean array for if the predicted direction matched the true direction
            - numpy.ndarray: (N_TIMEPOINTS x N_TIMEPOINTS)
    
    Notes:
         - first row is the current epoch with only one prediction from self.prediction_interval minutes ago
         - the final row is N_TIMEPOINTS epochs ago with  (N_TIMEPOINTS x self.prediction_interval = 30min) minute predictions for the current timepoint
         - the final column is the current timepoint with various prediction distances (5min, 10min,...)
    """
    if response.prediction is None:
        return None, None
    elif len(response.prediction) != self.N_TIMEPOINTS:
        return None, None
    else:
        past_predictions = self.past_predictions[uid]
        prediction_array = np.concatenate((np.array(response.prediction).reshape(1,self.N_TIMEPOINTS), past_predictions), axis=0)
        close_price_array = np.repeat(np.array(close_price[1:]).reshape(1,self.N_TIMEPOINTS), self.N_TIMEPOINTS, axis=0)
        if len(past_predictions.shape) == 1:
            before_pred_vector = np.array([])
            before_close_vector = np.array([])
        else:
            # add the timepoint before the first t from past history for each epoch
            before_pred_vector = np.concatenate((prediction_array[1:,0], np.array([0]))).reshape(self.N_TIMEPOINTS+1, 1)
            before_close_vector = np.repeat(np.array(close_price[0]), self.N_TIMEPOINTS, axis=0).reshape(self.N_TIMEPOINTS, 1)
        # take the difference between timepoints and remove the oldest prediction epoch (it is now obselete)
        pred_dir = (before_pred_vector - prediction_array)[:-1,:]
        close_dir = (before_close_vector - close_price_array)
        correct_dirs = time_shift((close_dir>=0)==(pred_dir>=0))
        deltas = np.abs(time_shift(close_price_array)-time_shift(prediction_array[:-1,:]))
        return deltas, correct_dirs
        
def rank_miners_by_epoch(deltas: np.ndarray, correct_dirs: np.ndarray):
    """
    Generates the rankings for each miner (rows) first according to their correct direction (bool), then by delta (float)

    Args:
        deltas (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint (e.g. the 5min prediction)
        correct_dirs (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint

    Returns:
        all_ranks (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint
    """
    correct_deltas = np.full(deltas.shape, np.nan)
    correct_deltas[correct_dirs] = deltas[correct_dirs]
    incorrect_deltas = np.full(deltas.shape, np.nan)
    incorrect_deltas[~correct_dirs] = deltas[~correct_dirs]
    correct_ranks = rank_columns(correct_deltas)
    incorrect_ranks = rank_columns(incorrect_deltas)+np.nanmax(correct_ranks, axis=0)
    all_ranks = correct_ranks
    all_ranks[~correct_dirs] = incorrect_ranks[~correct_dirs]
    return all_ranks

def rank_columns(array):
    """
    Changes the values of array into within-column ranks, putting nans at the lowest ranks

    Args:
        array (numpy.ndarray): a 2D array of values

    Returns:
        ranked_array (numpy.ndarray): array where the values are replaced with within-column rank
    """
    ranked_array = np.copy(array)
    # Iterate over each column
    for col in range(array.shape[1]):
        # Extract the column
        col_data = array[:, col]
        # Get indices of non-NaN values
        non_nan_indices = ~np.isnan(col_data)
        # Extract non-NaN values and sort them
        non_nan_values = col_data[non_nan_indices]
        sorted_indices = np.argsort(non_nan_values)
        ranks = np.empty_like(non_nan_values)
        # Assign ranks
        ranks[sorted_indices] = np.arange(1, len(non_nan_values) + 1)
        # Place ranks back into the original column, preserving NaNs
        ranked_array[non_nan_indices, col] = ranks
    return ranked_array

def time_shift(array):
    """
    This function alligns the timepoints of past_predictions with the current epoch
    and replaces predictions that havent come to fruition with nans.

    Args:
        array (np.ndarray): a square matrix

    Returns:
        shifted_array (np.ndarray): a square matrix where the diagonal elements become the last column,
            the unfulfilled predictions are removed and filled with nans

    Example:
        >>> test_array = np.array([[0,5,10,15,20,25], # - response.prediction on the current timepoint
                            [-5,0,5,10,15,20],
                            [-10,-5,0,5,10,15],
                            [-15,-10,-5,0,5,10],
                            [-20,-15,-10,-5,0,5],  
                            [-25,-20,-15,-10,-5,0], # - 30 minute prediction for time 0
                            [-30,-25,-20,15,-10,-5]])  # - the obseleted prediction
        >>> shifted_array =time_shift(test_array)
        >>> print(shifted_array)
    """
    shifted_array = np.full((array.shape[0], array.shape[1]), np.nan)
    for i in range(array.shape[0]):
        if i != range(array.shape[0]):
            shifted_array[i,-i-1:] = array[i,0:i+1]
        else:
            shifted_array[i,:] = array[i,:]
    return shifted_array

def update_synapse(self, uid, response: Challenge):
    """
    This is an example of Google style.

    Args:
        uid (int): The miner uid taken from the metagraph to be updated
        response (Challenge): The synapse response from the miner containing the prediction

    Returns:
        changes the value of self.past_predictions[uid] to include the most recent prediction and remove the oldest prediction
    """
    past_predictions = self.past_predictions[uid]
    new_past_predictions = np.concatenate((np.array(response.prediction).reshape(1,6), past_predictions), axis=0)
    self.past_predictions[uid] = new_past_predictions[0:-1,:] # remove the oldest epoch


################################################################################
#                                Main Function                                 #
################################################################################
def get_rewards(
    self,
    responses: List[Challenge],
    miner_uids: List[int],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[Challenge]): A list of responses from the miner.
    
    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    N_TIMEPOINTS = self.N_TIMEPOINTS
    prediction_interval = self.prediction_interval
    if len(responses) == 0:
        bt.logging.info("Got no responses. Returning reward tensor of zeros.")
        return [], torch.zeros_like(0).to(self.device)  # Fallback strategy: Log and return 0.

    # Prepare to extract close price for this timestamp
    ticker_symbol = '^GSPC'
    ticker = yf.Ticker(ticker_symbol)

    timestamp = responses[0].timestamp
    timestamp = datetime.fromisoformat(timestamp)

    # Round up current timestamp and then wait until that time has been hit
    rounded_up_time = timestamp - timedelta(minutes=timestamp.minute % prediction_interval,
                                    seconds=timestamp.second,
                                    microseconds=timestamp.microsecond) + timedelta(minutes=prediction_interval, seconds=1)
    
    ny_timezone = timezone('America/New_York')

    while (datetime.now(ny_timezone) < rounded_up_time):
        bt.logging.info(f"Waiting for next {prediction_interval}m interval...")
        if(datetime.now(ny_timezone).minute%20==0):
            self.resync_metagraph()
        time.sleep(15)

    
    data = yf.download(tickers=ticker_symbol, period='5d', interval='5m', progress=False)
    #bt.logging.info("Procured data from yahoo finance.")
    # add an extra timepoint for dir_acc calculation
    bt.logging.info(data.iloc[(-N_TIMEPOINTS-1):])
    close_price = data['Close'].iloc[(-N_TIMEPOINTS-1):].tolist()
    close_price_revealed = ' '.join(str(price) for price in close_price)

    bt.logging.info(f"Revealing close prices for this interval: {close_price_revealed}")

    # Preallocate an array (nMiners x N_TIMEPOINTS x N_TIMEPOINTS) where the third dimension is t-1, t-2,...,t-N_TIMEPOINTS for past predictions
    raw_deltas = np.full((len(responses),N_TIMEPOINTS,N_TIMEPOINTS), np.nan)
    raw_correct_dir = np.full((len(responses),N_TIMEPOINTS,N_TIMEPOINTS), False)
    ranks = np.full((len(responses),N_TIMEPOINTS,N_TIMEPOINTS), np.nan)
    for x,response in enumerate(responses):
        # calc_raw also does many helpful things like shifting epoch to 
        delta , correct = calc_raw(self, miner_uids[x], response, close_price)
        if delta is None or correct is None:
            if response.prediction is None:
                # no response generated
                bt.logging.info(f'Netuid {x} returned no response. Setting incentive to 0')
                raw_deltas[x,:,:], raw_correct_dir[x,:,:] = np.nan, np.nan
            else:
                # wrong size response generated
                bt.logging.info(f'Netuid {x} returned {len(response.prediction)} predictions instead of {N_TIMEPOINTS}. Setting incentive to 0')
                raw_deltas[x,:,:], raw_correct_dir[x,:,:] = np.nan, np.nan
            continue
        else:
            raw_deltas[x,:,:] = delta
            raw_correct_dir[x,:,:] = correct
        update_synapse(self, miner_uids[x], response, close_price)

    # raw_deltas is now a full of the last N_TIMEPOINTS of prediction deltas, same for raw_correct_dir
    ranks = np.full((len(responses),N_TIMEPOINTS,N_TIMEPOINTS), np.nan)
    for t in range(N_TIMEPOINTS):
        ranks[:,:,t] = rank_miners_by_epoch(raw_deltas[:,:,t], raw_correct_dir[:,:,t])

    incentives = np.nanmean(np.nanmean(ranks, axis=2), axis=1).argsort().argsort()
    reward = np.exp(-0.05*incentives)
    reward[incentives>100] = 0
    reward = reward/np.max(reward)
    return torch.FloatTensor(reward)
