# General imports
import bittensor as bt

# setup_wandb
import wandb
import os
import pathlib

# Is valid time
from datetime import datetime, timedelta
from pytz import timezone
import pandas_market_calendars as mcal
from numpy import full, nan

import argparse


def setup_wandb(self):
    netrc_path = pathlib.Path.home() / ".netrc"
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

def market_is_open():
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
    est = timezone('America/New_York')
    now = datetime.now(est)
    # Check if today is Monday through Friday
    if now.weekday() >= 5:  # 0 is Monday, 6 is Sunday
        return False
    # Check if the NYSE is open (i.e. not a holiday)
    result = mcal.get_calendar("NYSE").schedule(start_date=now, end_date=now)
    if result.empty == True:
        return False
    # Check if the current time is between 9:30 AM and 4:00 PM
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if not (start_time <= now <= end_time):
        return False
    # if all checks pass, return true
    return True

def is_query_time(prediction_interval, timestamp):
    now_ts = datetime.now(timezone('America/New_York')).timestamp()
    open_ts = datetime.now(timezone('America/New_York')).replace(hour=9, minute=30, second=0, microsecond=0).timestamp()
    sec_since_open = now_ts - open_ts
    # if it is within 120 seconds of the start of the prediction epoch and at least prediction_interval minutes have passed, return true
    result = sec_since_open % (prediction_interval*60) < 120 and datetime.now(timezone('America/New_York')) - datetime.fromisoformat(timestamp) > timedelta(minutes=prediction_interval)
    return result

def print_info(self):
    metagraph = self.metagraph
    self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
    log = (
        "Validator | "
        f"UID:{self.my_uid} | "
        f"Block:{self.current_block} | "
        f"Stake:{metagraph.S[self.my_uid]} | "
        f"VTrust:{metagraph.Tv[self.my_uid]} | "
        f"Dividend:{metagraph.D[self.my_uid]} | "
        f"Emission:{metagraph.E[self.my_uid]}"
    )
    bt.logging.info(log)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Validator Configuration")
    parser.add_argument("--subtensor.chain_endpoint", type=str, default="wss://entrypoint-finney.opentensor.ai:443") #for testnet: wss://test.finney.opentensor.ai:443
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--netuid", type=int, default=28)
    parser.add_argument("--neuron.name", type=str, default='validator')
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument("--logging.level", choices=['info', 'debug', 'trace'], default='info')
    parser.add_argument("--logging.logging_dir", type=str, default='~/.bittensor/validators')
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--prediction_interval", type=int, default=5)
    parser.add_argument("--N_TIMEPOIONTS", type=int, default=6)
    return parser.parse_args()

def resync_metagraph(self):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")
    self.metagraph.sync(subtensor=self.subtensor)

    bt.logging.info(
        "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
    )
    # Zero out all hotkeys that have been replaced.
    for uid, hotkey in enumerate(self.hotkeys):
        if hotkey != self.metagraph.hotkeys[uid]:
            self.scores[uid] = 0  # hotkey has been replaced
            self.past_predictions[uid] = full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan) # reset past predictions

    # Check to see if the metagraph has changed size.
    # If so, we need to add new hotkeys and moving averages.
    if len(self.hotkeys) < len(self.metagraph.hotkeys):
        # Update the size of the moving average scores.
        new_moving_average = [1.0] * len(self.metagraph.S)
        min_len = min(len(self.hotkeys), len(self.scores))
        new_moving_average[:min_len] = self.scores[:min_len]
        self.scores = new_moving_average


def log_wandb(self, responses, rewards, miner_uids):
    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
            }
            for miner_uid, response, reward in zip(
                miner_uids, responses, rewards.tolist()
            )
        }
    }
    wandb.log(wandb_val_log)