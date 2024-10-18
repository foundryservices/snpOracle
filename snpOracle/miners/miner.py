import asyncio
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pytz import timezone

from snpOracle.protocol import Challenge
from snpOracle.utils import (
    Config,
    load_model,
    miner_blacklist,
    miner_priority,
    parse_arguments,
    predict,
    prep_data,
    print_info,
    scale_data,
    setup_bittensor_objects,
    verify,
)

# Use an executor to offload blocking operations
executor = ThreadPoolExecutor(max_workers=5)
load_dotenv()


class Miner:
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "Miner"

    def __init__(self, config=None):
        super().__init__(config=config)
        args = parse_arguments()
        config = Config(args)
        self.config = config
        setup_bittensor_objects(self.config)
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=miner_blacklist,
            priority_fn=miner_priority,
            verify_fn=(verify if not self.config.neuron.disable_verification else None),
        )
        # Pass the Axon information to the network
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.current_prediction = [datetime.now(timezone("America/New_York")), [0]]

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning("You are allowing non-validators to send requests to your miner. This is a security risk.")
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning("You are allowing non-registered entities to send requests to your miner. This is a security risk.")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

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

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while self.block - self.metagraph.last_update[self.uid] < self.config.neuron.epoch_length:
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
        self.metagraph.last_update[self.uid] = self.block

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

        # Always save state.
        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (self.block - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "Miner"  # don't set weights if you're a miner


# Run the miner
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            print_info()
            time.sleep(15)
