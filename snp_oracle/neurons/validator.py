import os
import pathlib
import time
from datetime import datetime
from typing import List

import bittensor as bt
import pandas_market_calendars as mcal
import pytz
import wandb
from dotenv import load_dotenv
from numpy import full, nan

from snp_oracle import __version__
from snp_oracle.predictionnet.base.validator import BaseValidatorNeuron
from snp_oracle.predictionnet.utils.huggingface import HfInterface
from snp_oracle.predictionnet.utils.uids import check_uid_availability
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
        self.hf_interface = HfInterface()
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
            if uid_is_available:
                self.past_predictions[uid] = full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)
        netrc_path = pathlib.Path.home() / ".netrc"
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            bt.logging.info("WANDB_API_KEY is set")
        bt.logging.info("~/.netrc exists:", netrc_path.exists())
        if wandb_api_key is None and not netrc_path.exists():
            bt.logging.warning("WANDB_API_KEY not found in environment variables.")

        wandb.init(
            project=f"sn{self.config.netuid}-validators",
            mode="disabled" if not getattr(self.config.neuron, "wandb_on", False) else "online",
            entity="foundryservices",
            config={
                "hotkey": self.wallet.hotkey.ss58_address,
            },
            name=f"validator-{self.uid}-{__version__}",
            resume="auto",
            dir=self.config.neuron.full_path,
            reinit=True,
        )

    async def is_valid_time(self):
        """
        This function checks if the NYSE is open and validators should send requests.
        The final valid time is 4:00 PM - prediction length (self.INTERVAL) so that the final prediction is for 4:00 PM

        Returns:
            True if the NYSE is open and the current time is between 9:30 AM and (4:00 PM - self.INTERVAL)
            False otherwise

        Notes:
        ------
        Timezone is set to America/New_York

        """
        est = pytz.timezone("America/New_York")
        now = datetime.now(est)
        # Check if today is Monday through Friday
        if now.weekday() >= 5:  # 0 is Monday, 6 is Sunday
            return False
        # Check if the NYSE is open (i.e. not a holiday)
        if not self.market_is_open(now):
            return False
        # Check if the current time is between 9:30 AM and 4:00 PM
        start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if not (start_time <= now <= end_time):
            return False
        # if all checks pass, return true
        return True

    def market_is_open(self, date):
        """
        This is an extra check for holidays where the NYSE is closed

        Args:
            date (datetime): The date to check

        Returns:
            True if the NYSE is open.
            False otherwise

        """
        result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
        return not result.empty

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
        return await forward(self)

    def confirm_models(self, responses, miner_uids) -> List[bool]:
        models_confirmed = []
        self.hf_interface.update_collection(responses)
        for response, uid in zip(responses, miner_uids):
            models_confirmed.append(self.hf_interface.hotkeys_match(response, self.metagraph.hotkeys[uid]))
        return models_confirmed

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Get all values in one go to avoid multiple concurrent requests
        try:
            current_block = self.block  # Single websocket call
            stake = float(metagraph.S[self.uid])
            vtrust = float(metagraph.Tv[self.uid])
            dividend = float(metagraph.D[self.uid])
            emission = float(metagraph.E[self.uid])

            log = (
                "Validator | "
                f"Step:{self.step} | "
                f"UID:{self.uid} | "
                f"Block:{current_block} | "
                f"Stake:{stake:.4f} | "
                f"VTrust:{vtrust:.4f} | "
                f"Dividend:{dividend:.4f} | "
                f"Emission:{emission:.4f}"
            )
            bt.logging.info(log)
        except Exception as e:
            bt.logging.error(f"Error getting validator info: {e}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            validator.print_info()
            time.sleep(15)
