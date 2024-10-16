import os
from pathlib import Path

import bittensor as bt
import wandb


def print_info(self) -> None:
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
