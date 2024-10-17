import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import bittensor as bt
import git
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import ta
import wandb
import yfinance as yf
from pandas import DataFrame
from pytz import timezone
from sklearn.preprocessing import MinMaxScaler


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


def print_info(self) -> None:
    metagraph = self.metagraph
    weight_timing = self.set_weights_rate - self.blocks_since_last_update + 1
    log = (
        "Validator | "
        f"UID:{self.my_uid} | "
        f"Block:{self.current_block} | "
        f"Stake:{metagraph.S[self.my_uid]:.3f} | "
        f"VTrust:{metagraph.Tv[self.my_uid]:.3f} | "
        f"Dividend:{metagraph.D[self.my_uid]:.3f} | "
        f"Emission:{metagraph.E[self.my_uid]:.3f} | "
        f"Seting weights in {weight_timing} blocks"
    )
    bt.logging.info(log)


def setup_wandb(self) -> None:
    netrc_path = Path.home() / ".netrc"
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        bt.logging.info("WANDB_API_KEY is set")
    bt.logging.info("~/.netrc exists:", netrc_path.exists())
    if wandb_api_key is None and not netrc_path.exists():
        bt.logging.warning("WANDB_API_KEY not found in environment variables.")
    wandb.init(
        project=f"sn{self.config.netuid}-validators",
        entity="foundryservices",
        config={
            "hotkey": self.wallet.hotkey.ss58_address,
        },
        name=f"validator-{self.my_uid}-{'0.0.1'}",
        resume="auto",
        dir=self.config.full_path,
        reinit=True,
    )


def log_wandb(responses, rewards, miner_uids):
    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
            }
            for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
        }
    }
    wandb.log(wandb_val_log)


def setup_logging(config):
    if config.logging.level == "trace":
        bt.logging.set_trace()
    elif config.logging.level == "debug":
        bt.logging.set_debug()
    else:
        # set to info by default
        pass
    bt.logging.info(f"Set logging level to {config.logging.level}")

    full_path = Path(f"~/.bittensor/validators/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator").expanduser()
    full_path.mkdir(parents=True, exist_ok=True)
    config.full_path = str(full_path)

    bt.logging.info(f"Arguments: {vars(config)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validator Configuration")
    parser.add_argument("--subtensor.chain_endpoint", type=str, default=None)  # for testnet: wss://test.finney.opentensor.ai:443
    parser.add_argument(
        "--subtensor.network",
        choices=["finney", "test", "local"],
        default="finney",
    )
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--netuid", type=int, default=28)
    parser.add_argument("--neuron.name", type=str, default="validator")
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument("--logging.level", choices=["info", "debug", "trace"], default="info")
    parser.add_argument("--autoupdate", action="store_true", dest="autoupdate")
    parser.add_argument("--logging.logging_dir", type=str, default="~/.bittensor/validators")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prediction_interval", type=int, default=5)
    parser.add_argument("--N_TIMEPOINTS", type=int, default=6)
    parser.add_argument("--vpermit_tao_limit", type=int, default=1024)
    parser.add_argument("--wandb_on", action="store_true", dest="wandb_on")
    parser.add_argument("--reset_state", action="store_true", dest="reset_state")
    return parser.parse_args(namespace=NestedNamespace())


class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if "." in name:
            group, name = name.split(".", 1)
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def get(self, key, default=None):
        if "." in key:
            group, key = key.split(".", 1)
            return getattr(self, group, NestedNamespace()).get(key, default)
        return self.__dict__.get(key, default)


def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_version() -> Optional[str]:
    repo = git.Repo(search_parent_directories=True)
    branch_name = repo.active_branch.name
    url = f"https://github.com/foundryservices/snpOracle/blob/{branch_name}/snpOracle/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("github api call failed")
        return None
    match = re.search(r"__version__ = (.{1,10})", response.text)

    version_match = re.search(r"\d+\.\d+\.\d+", match.group(1))
    if not version_match:
        raise Exception("Version information not found")

    return version_match.group()


class Config:
    def __init__(self, args):
        # Add command-line arguments to the Config object
        for key, value in vars(args).items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def prep_data(drop_na: bool = True) -> DataFrame:
    """
    Prepare data by calling Yahoo Finance SDK & computing technical indicators.

    The function gets the last 60 day data for the S&P 500 index at a 5m granularity
    and then computes the necessary technical indicators, resets index and drops rows
    with NA values if mentioned.

    Input:
        :param drop_na: The drop_na flag is used to tell the function whether to drop rows
                        with nan values or keep them.
        :type drop_na: bool

    Output:
        :returns: A pandas dataframe with the OHLCV data, along with the some technical indicators.
                  The dataframe also has the next close as a column to predict future close price using
                  current data.
        :rtype: pd.DataFrame
    """
    # Fetch S&P 500 data - when capturing data any interval, the max we can go back is 60 days
    # using Yahoo Finance's Python SDK
    data = yf.download("^GSPC", period="max", interval="5m")

    # Calculate technical indicators - all technical indicators computed here are based on the 5m data
    # For example - SMA_50, is not a 50-day moving average, but is instead a 50 5m moving average
    # since the granularity of the data we procure is at a 5m interval.
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["CCI"] = ta.trend.CCIIndicator(data["High"], data["Low"], data["Close"]).cci()
    data["Momentum"] = ta.momentum.ROCIndicator(data["Close"]).roc()
    for i in range(1, 7):
        data[f"NextClose{i}"] = data["Close"].shift(-1 * i)

    # Drop NaN values
    if drop_na:
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data


def scale_data(data: DataFrame) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Normalize the data procured from yahoo finance between 0 & 1

    Function takes a dataframe as an input, scales the input and output features and
    then returns the scaler itself, along with the scaled inputs and outputs. Scaler is
    returned to ensure that the output being predicted can be rescaled back to a proper
    value.

    Input:
        :param data: The S&P 500 data procured from a certain source at a 5m granularity
        :type data: pd.DataFrame

    Output:
        :returns: A tuple of 3 values -
                - scaler : which is the scaler used to scale the data (MixMaxScaler)
                - X_scaled : input/features to the model scaled (np.ndarray)
                - y_scaled : target variable of the model scaled (np.ndarray)
    """
    X = data[["Open", "High", "Low", "Volume", "SMA_50", "SMA_200", "RSI", "CCI", "Momentum"]].values

    # Prepare target variable
    y = data[["NextClose1", "NextClose2", "NextClose3", "NextClose4", "NextClose5", "NextClose6"]].values

    y = y.reshape(-1, 6)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return scaler, X_scaled, y_scaled


def predict(timestamp: datetime, scaler: MinMaxScaler, model, type, prediction_interval: int = 5) -> float:
    """
    Predicts the close price of the next 5m interval

    The predict function also ensures that the data is procured - using yahoo finance's python module,
    prepares the data and gets basic technical analysis metrics, and finally predicts the model
    and scales it based on the scaler used previously to create the model.

    Input:
        :param timestamp: The timestamp of the instant at which the request is sent by the validator
        :type timestamp: datetime.datetime

        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param model: The model used to make the predictions - in this case a .h5 file
        :type model: A keras model instance

    Output:
        :returns: The close price of the 5m period that ends at the timestamp passed by the validator
        :rtype: float
    """
    # calling this to get the data - the information passed by the validator contains
    # only a timestamp, it is on the miners to get the data and prepare is according to their requirements
    data = prep_data(drop_na=False)

    # Ensuring that the Datetime column in the data procured from yahoo finance is truly a datetime object
    data["Datetime"] = pd.to_datetime(data["Datetime"])

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    dt = datetime.fromisoformat(timestamp)
    pred_time = dt - timedelta(minutes=dt.minute % prediction_interval, seconds=dt.second, microseconds=dt.microsecond)

    matching_row = data[data["Datetime"] == pred_time]

    print(pred_time, matching_row)

    # Check if matching_row is empty
    if matching_row.empty:
        print("No matching row found for the given timestamp.")
        return 0.0

    # data.to_csv('mining_models/base_miner_data.csv')
    input = matching_row[["Open", "High", "Low", "Volume", "SMA_50", "SMA_200", "RSI", "CCI", "Momentum"]]

    if type != "regression":
        input = np.array(input, dtype=np.float32).reshape(1, -1)
        input = np.reshape(input, (1, 1, input.shape[1]))
        print(input)

    prediction = model.predict(input)
    if type != "regression":
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))

    print(prediction)
    return prediction
