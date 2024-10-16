import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bittensor as bt
import git
import pandas_market_calendars as mcal
import requests
import wandb
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


def print_info(self) -> None:
    metagraph = self.metagraph
    self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
    log = (
        "Validator | "
        f"UID:{self.my_uid} | "
        f"Block:{self.current_block} | "
        f"Stake:{metagraph.S[self.my_uid]:.3f} | "
        f"VTrust:{metagraph.Tv[self.my_uid]:.3f} | "
        f"Dividend:{metagraph.D[self.my_uid]:.3f} | "
        f"Emission:{metagraph.E[self.my_uid]:.3f} | "
        f"Seting weights in {(self.set_weights_rate-self.blockes_since_last_update)*12} seconds"
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
