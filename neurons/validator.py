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


import os
import pathlib
import time
from datetime import datetime

# Bittensor
import bittensor as bt
import pandas_market_calendars as mcal
import pytz
import wandb
from dotenv import load_dotenv
from numpy import full, nan

from predictionnet import __version__

# import base validator class which takes care of most of the boilerplate
from predictionnet.base.validator import BaseValidatorNeuron
from predictionnet.utils.uids import check_uid_availability

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
        self.INTERVAL = (
            self.prediction_interval * self.N_TIMEPOINTS
        )  # 30 Minutes
        self.past_predictions = {}
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(
                self.metagraph, uid, self.config.neuron.vpermit_tao_limit
            )
            if uid_is_available:
                self.past_predictions[uid] = full(
                    (self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan
                )
        netrc_path = pathlib.Path.home() / ".netrc"
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            bt.logging.info("WANDB_API_KEY is set")
        bt.logging.info("~/.netrc exists:", netrc_path.exists())
        if wandb_api_key is None and not netrc_path.exists():
            bt.logging.warning(
                "WANDB_API_KEY not found in environment variables."
            )

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
        result = mcal.get_calendar("NYSE").schedule(
            start_date=date, end_date=date
        )
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

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )

        log = (
            "Validator | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"VTrust:{metagraph.Tv[self.uid]} | "
            f"Dividend:{metagraph.D[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            validator.print_info()
            time.sleep(15)
