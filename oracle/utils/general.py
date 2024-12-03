import argparse
import asyncio
import re
from typing import Optional

import bittensor as bt
import git
import requests
from numpy import argsort, array, concatenate, copy, cumsum, empty_like, full, nan, nanmax, ndarray, repeat
from pandas import DataFrame

from oracle.utils.classes import NestedNamespace
from oracle.utils.timestamp import get_mature_timestamps, get_now, iso8601_to_datetime, round_minute_down


def parse_arguments(parser: Optional[argparse.ArgumentParser] = None):
    """Used to overwrite defaults when params are passed into the script.

    Args:
        parser (Optional[argparse.ArgumentParser], optional): _description_. Default arguments shown below.

    Example:
        >>> python3 -m start_miner.py --netuid 2
        >>> args = parse_arguments()
        >>> print(args.subtensor.chain_endpoint)

    Returns:
        namespace (NestedNamespace): Returns a nested arparse.namespace object which contains all the arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument(
        "--subtensor.chain_endpoint", type=str, default=None
    )  # for testnet: wss://test.finney.opentensor.ai:443
    parser.add_argument(
        "--subtensor.network",
        choices=["finney", "test", "local"],
        default="finney",
    )
    parser.add_argument("--wallet.name", type=str, default="default", help="Coldkey name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Hotkey name")
    parser.add_argument("--netuid", type=int, default=1, help="Subnet netuid")
    parser.add_argument("--neuron.name", type=str, default="validator", help="What to call this process")
    parser.add_argument(
        "--neuron.type",
        type=str,
        choices=["Validator", "Miner"],
        default="Validator",
        help="What type of neuron this is",
    )
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument("--logging.level", type=str, choices=["info", "debug", "trace"], default="info")
    parser.add_argument("--logging.logging_dir", type=str, default="~/.bittensor/validators")
    parser.add_argument(
        "--blacklist.force_validator_permit", action="store_false", dest="blacklist.force_validator_permit"
    )
    parser.add_argument("--blacklist.allow_non_registered", action="store_true", dest="blacklist.allow_non_registered")
    parser.add_argument("--autoupdate", action="store_true", dest="autoupdate")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prediction_interval", type=int, default=5)
    parser.add_argument("--N_TIMEPOINTS", type=int, default=6)
    parser.add_argument("--vpermit_tao_limit", type=int, default=1024)
    parser.add_argument("--wandb_on", action="store_true", dest="wandb_on")
    parser.add_argument("--reset_state", action="store_true", dest="reset_state", help="Overwrites the state file")
    parser.add_argument("--timeout", type=int, default=16, help="allowable nonce delay time (seconds)")
    parser.add_argument("--print_cadence", type=float, default=12, help="how often to print stats (seconds)")
    parser.add_argument("--forward_function", type=str, default="forward", help="name of the forward function to use")
    return parser.parse_args(namespace=NestedNamespace())


def get_version() -> Optional[str]:
    """Pulls the version of the precog-subnet repo from GitHub.

    Returns:
        Optional[str]: Repo version in the format: '1.1.0'
    """
    repo = git.Repo(search_parent_directories=True)
    branch_name = repo.active_branch.name
    url = f"https://github.com/foundryservices/snpOracle/blob/{branch_name}/snpOracle/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("Failed to get version from GitHub")
        return None
    match = re.search(r"__version__ = (.{1,10})", response.text)

    version_match = re.search(r"\d+\.\d+\.\d+", match.group(1))
    if not version_match:
        raise Exception("Version information not found")

    return version_match.group()


def convert_predictions_to_matrix(prediction_dict, timestamp: None):
    if timestamp is None:
        now = round_minute_down(get_now())
    else:
        now = iso8601_to_datetime(timestamp)
    timestamps_to_pull = get_mature_timestamps(now, interval=-5)
    predictions = full((len(timestamps_to_pull), len(timestamps_to_pull)), nan)
    for i, timestamp in enumerate(timestamps_to_pull):
        try:
            predictions[i, :] = prediction_dict[timestamp]
        except KeyError:
            predictions[i, :] = nan
    return predictions, timestamps_to_pull


def convert_prices_to_matrix(price_dict, reference_timestamp: None, N_TIMEPOINTS: int = 6):
    if reference_timestamp is None:
        now = round_minute_down(get_now())
    else:
        now = iso8601_to_datetime(reference_timestamp)
    timestamps_to_pull = get_mature_timestamps(now, interval=-5, N_TIMEPOINTS=N_TIMEPOINTS + 1)
    prices = full(len(timestamps_to_pull), nan)
    for i, timestamp in enumerate(timestamps_to_pull):
        try:
            prices[i] = price_dict[timestamp]
        except KeyError:
            prices[i] = nan
    prices = prices[::-1]
    price_array = repeat(
        array(prices[1:]).reshape(1, N_TIMEPOINTS),
        N_TIMEPOINTS,
        axis=0,
    )
    return prices, price_array


def time_shift(array: ndarray) -> ndarray:
    """
    This function alligns the timepoints of past_predictions with the current epoch
    and replaces predictions that havent come to fruition with nans.

    Args:
        array (np.ndarray): a square matrix

    Returns:
        shifted_array (np.ndarray): a square matrix where the diagonal elements become the last column,
            the unfulfilled predictions are removed and filled with nans

    Example:
        >>> test_array = np.array([[0,5,10,15,20,25], # - response.prediction on the current timepoint (requested 5 minutes ago)
                                   [-5,0,5,10,15,20], # - 10 minute prediction for time 0
                                   [-10,-5,0,5,10,15],
                                   [-15,-10,-5,0,5,10],
                                   [-20,-15,-10,-5,0,5],
                                   [-25,-20,-15,-10,-5,0], # - 30 minute prediction for time 0
                                   [-30,-25,-20,15,-10,-5]])  # - the obseleted prediction
        >>> shifted_array =time_shift(test_array)
        >>> print(shifted_array)
    """
    shifted_array = full((array.shape[0], array.shape[1]), nan)
    for i in range(array.shape[0]):
        if i != range(array.shape[0]):
            shifted_array[i, -i - 1 :] = array[i, 0 : i + 1]
        else:
            shifted_array[i, :] = array[i, :]
    return shifted_array


def pd_to_dict(data: DataFrame) -> dict:
    price_dict = {}
    for i in range(len(data)):
        price_dict[data.index[i]] = data.iloc[i]["Close"].values.item()
    return price_dict


def rank_miners_by_epoch(deltas: ndarray, correct_dirs: ndarray) -> ndarray:
    """
    Generates the rankings for each miner (rows) first according to their correct_dirs (bool), then by deltas (float)

    Args:
        deltas (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint (e.g. the 5min prediction)
        correct_dirs (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint

    Returns:
        all_ranks (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint
    """
    correct_deltas = full(deltas.shape, nan)
    correct_deltas[correct_dirs] = deltas[correct_dirs]
    incorrect_deltas = full(deltas.shape, nan)
    incorrect_deltas[~correct_dirs] = deltas[~correct_dirs]
    correct_ranks = rank_columns(correct_deltas)
    incorrect_ranks = rank_columns(incorrect_deltas) + nanmax(correct_ranks, axis=0)
    all_ranks = correct_ranks
    all_ranks[~correct_dirs] = incorrect_ranks[~correct_dirs]
    return all_ranks - 1  # zero indexing


def rank_columns(array: ndarray) -> ndarray:
    """
    Changes the values of array into within-column ranks, preserving nans

    Args:
        array (numpy.ndarray): a 2D array of values

    Returns:
        ranked_array (numpy.ndarray): array where the values are replaced with within-column rank
    """
    ranked_array = copy(array)
    # Iterate over each column
    for col in range(array.shape[1]):
        ranked_array[:, col] = rank(array[:, col])
    return ranked_array


def rank(vector):
    if vector is None or len(vector) <= 1:
        return array([0])
    else:
        # Sort the array and get the indices that would sort it
        sorted_indices = argsort(vector)
        sorted_vector = vector[sorted_indices]
        # Create a mask for where each new unique value starts in the sorted array
        unique_mask = concatenate(([True], sorted_vector[1:] != sorted_vector[:-1]))
        # Use cumulative sum of the unique mask to get the ranks, then assign back in original order
        ranks = cumsum(unique_mask) - 1
        rank_vector = empty_like(vector, dtype=int)
        rank_vector[sorted_indices] = ranks
        return rank_vector


async def loop_handler(self, func, sleep_time=120):
    try:
        while not self.stop_event.is_set():
            async with self.lock:
                await func()
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        bt.logging.error(f"{func.__name__} cancelled")
        raise
    except KeyboardInterrupt:
        raise
    except Exception as e:
        bt.logging.error(f"{func.__name__} raised error: {e}")
        raise
    finally:
        async with self.lock:
            self.stop_event.set()
