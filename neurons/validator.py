# The MIT License (MIT)

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

import asyncio
import os
import pathlib
import pickle
import time
from typing import List

# Bittensor
import bittensor as bt
import wandb
from dotenv import load_dotenv

from predictionnet import __version__

# import base validator class which takes care of most of the boilerplate
from predictionnet.base.validator import BaseValidatorNeuron
from predictionnet.utils.bittensor import get_available_uids, print_info
from predictionnet.utils.classes import MinerHistory
from predictionnet.utils.huggingface import HfInterface
from predictionnet.utils.timestamp import elapsed_seconds, get_before, get_now, is_query_time, market_is_open

# Bittensor Validator Template:
from predictionnet.validator import forward

load_dotenv()


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        # basic params
        self.prediction_interval = 5  # in minutes
        self.N_TIMEPOINTS = 6  # number of timepoints to predict
        self.INTERVAL = self.prediction_interval * self.N_TIMEPOINTS  # 30 Minutes
        self.available_uids = asyncio.run(get_available_uids(self))
        self.hf_interface = HfInterface()
        self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
        self.timestamp = get_before(minutes=60)
        if self.config.reset_state:
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
            self.timestamp = get_before(minutes=60)
            self.save_state()
        else:
            self.load_state()

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
            name=f"validator-{self.uid}-{__version__}",
            resume="auto",
            dir=self.config.neuron.full_path,
            reinit=True,
        )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        if market_is_open():
            query_lag = elapsed_seconds(get_now() - self.timestamp)
            if (
                is_query_time(self.prediction_interval, self.timestamp)
                or query_lag >= 60 * self.prediction_interval
            ):
                await forward(self)
            else:
                print_info(self, "Market Open")
        else:
            print_info(self, "Market Closed")
        return

    def confirm_models(self, responses) -> List[bool]:
        models_confirmed = []
        self.hf_interface.update_collection(responses)
        for response, uid in zip(responses, self.available_uids):
            models_confirmed.append(
                self.hf_interface.hotkeys_match(response, self.metagraph.hotkeys[uid])
            )
        return models_confirmed

    def save_state(self):
        """Saves the state of the validator to a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")
        state = {
            "scores": self.scores,
            "MinerHistory": self.MinerHistory,
            "moving_average_scores": self.moving_average_scores,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        bt.logging.info(f"Saved {self.config.neuron.name} state.")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state_path = os.path.join(self.config.full_path, "state.pt")
        bt.logging.info(f"State path: {state_path}")
        if not os.path.exists(state_path):
            bt.logging.info("Skipping state load due to missing state.pt file.")
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
            self.timestamp = get_before(minutes=60)
            return
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self.scores = state["scores"]
            self.MinerHistory = state["MinerHistory"]
            self.moving_average_scores = state["moving_average_scores"]
            all_dates = [mh.latest_timestamp() for mh in self.MinerHistory.values()]
            self.timestamp = max(all_dates)
        except Exception as e:
            bt.logging.error(f"Failed to load state with error: {e}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(15)
