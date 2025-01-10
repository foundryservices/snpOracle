import asyncio
import threading
import time
import traceback

import bittensor as bt

from snp_oracle.predictionnet.base.neuron import BaseNeuron

# from substrateinterface import Keypair


# from snp_oracle.predictionnet.protocol import Challenge


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
            # verify_fn=self.verify,
        )
        bt.logging.info(f"Axon created: {self.axon}")
        self.nonces = {}
        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

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
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
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

    # def _to_seconds(self, nano: int) -> int:
    #     return nano / 1_000_000_000

    # async def verify(self, synapse: Challenge) -> None:
    #     """
    #     Verifies the authenticity and validity of incoming requests.

    #     Args:
    #         synapse: The Challenge object containing request details and credentials

    #     Raises:
    #         Exception: If verification fails due to missing nonce, invalid timestamps,
    #                 duplicate nonces, or signature mismatch
    #     """
    #     bt.logging.debug(f"checking nonce: {synapse.dendrite}")

    #     # Skip verification if no dendrite info
    #     if synapse.dendrite is None:
    #         return

    #     # Build keypair and verify signature
    #     keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

    #     # Check for missing nonce
    #     if synapse.dendrite.nonce is None:
    #         raise Exception("Missing Nonce")

    #     # Build unique endpoint identifier
    #     endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

    #     # Check timestamp validity
    #     cur_time = time.time_ns()
    #     allowed_delta = self.config.timeout * 1_000_000_000  # nanoseconds
    #     latest_allowed_nonce = synapse.dendrite.nonce + allowed_delta

    #     # Log timing details for debugging
    #     bt.logging.debug(f"synapse.dendrite.nonce: {synapse.dendrite.nonce}")
    #     bt.logging.debug(f"latest_allowed_nonce: {latest_allowed_nonce}")
    #     bt.logging.debug(f"cur time: {cur_time}")
    #     bt.logging.debug(f"delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}")

    #     # Verify nonce timing
    #     if self.nonces.get(endpoint_key) is None and synapse.dendrite.nonce > latest_allowed_nonce:
    #         raise Exception(
    #             f"Nonce is too old. Allowed delta in seconds: {self._to_seconds(allowed_delta)}, "
    #             f"got delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}"
    #         )

    #     # Verify nonce order
    #     if self.nonces.get(endpoint_key) is not None and synapse.dendrite.nonce <= self.nonces[endpoint_key]:
    #         raise Exception(
    #             f"Nonce is too small, already have a newer nonce in the nonce store, "
    #             f"got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}"
    #         )

    #     # Build and verify signature message
    #     message = (
    #         f"{synapse.dendrite.nonce}."
    #         f"{synapse.dendrite.hotkey}."
    #         f"{self.wallet.hotkey.ss58_address}."
    #         f"{synapse.dendrite.uuid}."
    #         f"{synapse.computed_body_hash}"
    #     )

    #     if not keypair.verify(message, synapse.dendrite.signature):
    #         raise Exception(
    #             f"Signature mismatch with {message} and {synapse.dendrite.signature}, "
    #             f"from hotkey {synapse.dendrite.hotkey}"
    #         )

    #     # Store nonce after successful verification
    #     self.nonces[endpoint_key] = synapse.dendrite.nonce  # type: ignore
