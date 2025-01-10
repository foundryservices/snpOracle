import asyncio
import os
import pathlib
import pickle
import time
from datetime import datetime
from typing import List

import bittensor as bt
import wandb
from dotenv import load_dotenv

from snp_oracle import __version__
from snp_oracle.predictionnet.base.validator import BaseValidatorNeuron
from snp_oracle.predictionnet.utils.dataset_manager import DatasetManager
from snp_oracle.predictionnet.utils.huggingface import HfInterface
from snp_oracle.predictionnet.utils.bittensor import get_available_uids, print_info
from snp_oracle.predictionnet.utils.classes import MinerHistory
from snp_oracle.predictionnet.utils.timestamp import elapsed_seconds, get_before, get_now, is_query_time, market_is_open
from snp_oracle.predictionnet.validator import forward

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
        self.past_predictions = {}
        self.available_uids = asyncio.run(get_available_uids(self))
        self.hf_interface = HfInterface()
        self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
        self.DatasetManager = DatasetManager(organization=self.config.neuron.organization)
        self.timestamp = get_before(minutes=60)
        self.first_closed_call = True # handles market close events
        if self.config.reset_state:
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
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
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        if market_is_open():
            if not self.first_closed_call:
                self.first_closed_call = True
            query_lag = elapsed_seconds(get_now() - self.timestamp)
            if is_query_time(self.prediction_interval, self.timestamp) or query_lag >= 60 * self.prediction_interval:
                await forward(self)
            else:
                print_info(self, "Market Open")
                asyncio.sleep(12)
        else:
            if self.first_closed_call:
                self.first_closed_call = False
                await self.handle_market_close(self.DatasetManager)
            print_info(self, "Market Closed")
            asyncio.sleep(120)
        return

    def confirm_models(self, responses) -> List[bool]:
        models_confirmed = []
        self.hf_interface.update_collection(responses)
        for response, uid in zip(responses, self.available_uids):
            models_confirmed.append(self.hf_interface.hotkeys_match(response, self.metagraph.hotkeys[uid]))
        return models_confirmed

    async def handle_market_close(self, dataset_manager: DatasetManager) -> None:
        """Handle data management operations when market is closed."""
        try:
            # Clean up old data
            dataset_manager.cleanup_local_storage(days_to_keep=2)

            # Upload today's data
            success, result = dataset_manager.batch_upload_daily_data()
            if success:
                bt.logging.success(
                    f"Daily batch upload completed. Uploaded {result.get('files_uploaded', 0)} files "
                    f"with {result.get('total_rows', 0)} rows to {result.get('repo_id')}"
                )
            else:
                bt.logging.error(f"Daily batch upload failed: {result.get('error')}")

        except Exception as e:
            bt.logging.error(f"Error during market close operations: {str(e)}")

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
            validator.print_info()
            time.sleep(15)
