import asyncio
import os
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bittensor as bt
import yfinance as yf
from dotenv import load_dotenv
from pytz import timezone

import predictionnet
from predictionnet.base.miner import BaseMinerNeuron

# Use an executor to offload blocking operations
executor = ThreadPoolExecutor(max_workers=5)
load_dotenv()


class Miner(BaseMinerNeuron):
    """
    Optimized Miner to handle ultra-fast requests with low latency.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.model_loc = self.config.model
        self.current_prediction = [datetime.now(timezone("America/New_York")), [0]]
        if self.config.neuron.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

    # Async function to avoid blocking I/O operations
    async def download_data(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, yf.download, "^GSPC", "5d", "1m", False)

    async def fit_model(self, y):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._fit_arima, y)

    async def forward(self, synapse: predictionnet.protocol.Challenge) -> predictionnet.protocol.Challenge:
        """
        Optimized forward function for low latency and caching.
        """
        bt.logging.info(f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}")

        timestamp = synapse.timestamp
        time_diff = datetime.fromisoformat(timestamp) - self.current_prediction[0]

        if time_diff.seconds < 180 and self.current_prediction[1][0] != 0:
            # Serve cached prediction if less than 3 minutes old
            synapse.prediction = self.current_prediction[1]
            self.current_prediction[0] = datetime.fromisoformat(timestamp)
        else:
            # Async call to download data and fit the model to avoid blocking
            data = await self.download_data()
            y = data["Close"].values[-500:]  # Only use the latest 500 data points

            model_fit = await self.fit_model(y)
            forecast = model_fit.forecast(steps=30)
            forecast = forecast[[4, 9, 14, 19, 24, 29]]  # Pick specific points from forecast

            self.current_prediction = [datetime.fromisoformat(timestamp), forecast.tolist()]
            synapse.prediction = forecast.tolist()

        if synapse.prediction:
            bt.logging.success(f"Predicted price 🎯: {synapse.prediction}")
        else:
            bt.logging.info("No price predicted for this request.")

        return synapse

    def save_state(self):
        pass

    def load_state(self):
        pass

    def print_info(self):
        """Print miner status information."""
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        log = (
            "Miner | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]} | "
            f"Incentive:{metagraph.I[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)

    async def blacklist(self, synapse: predictionnet.protocol.Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (predictionnet.protocol.Challenge): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.config.blacklist.allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from un-registered entities.
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")

    async def priority(self, synapse: predictionnet.protocol.Challenge) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (predictionnet.protocol.Challenge): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        prirority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority)
        return prirority


# Run the miner
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.print_info()
            time.sleep(15)
