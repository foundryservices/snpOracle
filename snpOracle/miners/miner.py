import asyncio
import os
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pytz import timezone

from snpOracle.protocol import Challenge
from snpOracle.utils import Config, load_model, parse_arguments, predict, prep_data, scale_data, setup_logging

# Use an executor to offload blocking operations
executor = ThreadPoolExecutor(max_workers=5)
load_dotenv()


class Miner:
    """
    Optimized Miner to handle ultra-fast requests with low latency.
    """

    def __init__(self, config=None):
        args = parse_arguments()
        config = Config(args)
        self.config = config
        self.model_loc = self.config.model
        setup_logging(self.config)
        self.current_prediction = [datetime.now(timezone("America/New_York")), [0]]
        if self.config.neuron.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

    # Async function to avoid blocking I/O operations
    async def download_data(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, prep_data())

    async def fit_model(self, y):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._fit_arima, y)

    async def forward(self, synapse: Challenge) -> Challenge:
        """
        Optimized forward function for low latency and caching.
        """
        bt.logging.info(f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}")

        timestamp = synapse.timestamp
        # Download the file
        if self.config.hf_repo_id == "LOCAL":
            model_path = f"./{self.config.model}"
            bt.logging.info(f"Model weights file from a local folder will be loaded - Local weights file path: {self.config.model}")
        else:
            if not os.getenv("HF_ACCESS_TOKEN"):
                print("Cannot find a Huggingface Access Token - model download halted.")
            token = os.getenv("HF_ACCESS_TOKEN")
            model_path = hf_hub_download(repo_id=self.config.hf_repo_id, filename=self.config.model, use_auth_token=token)
            bt.logging.info(f"Model downloaded from huggingface at {model_path}")

        model = load_model(model_path)
        data = self.download_data()
        scaler, _, _ = scale_data(data)

        # type needs to be changed based on the algo you're running
        # any algo specific change logic can be added to predict function in predict.py
        prediction = predict(timestamp, scaler, model, type="lstm")
        bt.logging.info(f"Prediction: {prediction}")
        # pred_np_array = np.array(prediction).reshape(-1, 1)

        # logic to ensure that only past 20 day context exists in synapse
        synapse.prediction = list(prediction[0])

        if synapse.prediction is None:
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

    async def blacklist(self, synapse: Challenge) -> typing.Tuple[bool, str]:
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

    async def priority(self, synapse: Challenge) -> float:
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
